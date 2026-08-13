"""Fatigue rule persistence and compliance administration routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import re
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from fatigue_engine import CUSTOM_FATIGUE_RULE_TYPES, SYSTEM_FATIGUE_RULES


@dataclass(frozen=True)
class FatigueRuleConfigDependencies:
    db: Any
    RosterSetting: Any
    current_unit_id: Callable[[], int]


def create_fatigue_rule_config_dependencies(
    *, db: Any, operational_models: Any, **services: Any
) -> FatigueRuleConfigDependencies:
    """Bind fatigue configuration records at the fatigue-policy boundary."""
    return FatigueRuleConfigDependencies(
        db=db,
        RosterSetting=operational_models.RosterSetting,
        **services,
    )


class FatigueRuleConfigService:
    """Load and persist airport-scoped fatigue rule configuration."""

    def __init__(self, dependencies: FatigueRuleConfigDependencies) -> None:
        self.dependencies = dependencies

    def load(self, unit_id: int | None = None) -> dict:
        resolved_unit_id = int(
            unit_id or self.dependencies.current_unit_id() or 1
        )
        system = {
            item["code"]: {
                **item,
                "parameters": {
                    key: dict(parameter)
                    for key, parameter in item["parameters"].items()
                },
                "enabled": True,
            }
            for item in SYSTEM_FATIGUE_RULES
        }
        custom = []
        definitions = {
            "early_start_before": "06:30",
            "night_period_start": "01:30",
            "night_period_end": "05:30",
        }
        row = self.dependencies.RosterSetting.query.filter_by(
            unit_id=resolved_unit_id, key="fatigue_rule_config"
        ).first()
        if row and row.value:
            try:
                saved = json.loads(row.value)
                for key in definitions:
                    candidate = str(
                        (saved.get("definitions") or {}).get(key) or ""
                    )
                    try:
                        datetime.strptime(candidate, "%H:%M")
                        definitions[key] = candidate
                    except ValueError:
                        pass
                for code, overrides in (saved.get("system") or {}).items():
                    if code in system and isinstance(overrides, dict):
                        system[code].update({
                            "name": str(
                                overrides.get("name") or system[code]["name"]
                            )[:120],
                            "severity": (
                                overrides.get("severity")
                                if overrides.get("severity")
                                in {"warning", "critical"}
                                else system[code]["severity"]
                            ),
                            "enabled": bool(overrides.get("enabled", True)),
                        })
                        saved_parameters = overrides.get("parameters") or {}
                        for key, parameter in system[code]["parameters"].items():
                            try:
                                value = float(
                                    saved_parameters.get(key, parameter["value"])
                                )
                                if value > 0:
                                    parameter["value"] = value
                            except (TypeError, ValueError):
                                pass
                for rule in saved.get("custom") or []:
                    if (
                        isinstance(rule, dict)
                        and rule.get("rule_type") in CUSTOM_FATIGUE_RULE_TYPES
                    ):
                        custom.append(rule)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return {
            "system": system,
            "custom": custom,
            "definitions": definitions,
        }

    def save(self, config: dict) -> None:
        unit_id = self.dependencies.current_unit_id()
        row = self.dependencies.RosterSetting.query.filter_by(
            unit_id=unit_id, key="fatigue_rule_config"
        ).first()
        if not row:
            row = self.dependencies.RosterSetting(
                unit_id=unit_id, key="fatigue_rule_config"
            )
            self.dependencies.db.session.add(row)
        row.value = json.dumps({
            "system": {
                code: {
                    "name": item["name"],
                    "severity": item["severity"],
                    "enabled": item["enabled"],
                    "parameters": {
                        key: parameter["value"]
                        for key, parameter in item["parameters"].items()
                    },
                }
                for code, item in config["system"].items()
            },
            "custom": config["custom"],
            "definitions": config["definitions"],
        }, sort_keys=True)
        self.dependencies.db.session.commit()


def compliance_month(ym: str | None) -> tuple[int, int]:
    today = date.today()
    value = (ym or f"{today.year:04d}-{today.month:02d}").strip()
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        abort(400, "Month must use YYYY-MM.")
    year, month = map(int, value.split("-"))
    if month not in range(1, 13):
        abort(400, "Invalid month.")
    return year, month


@dataclass(frozen=True)
class FatigueComplianceDependencies:
    db: Any
    Unit: Any
    is_admin_user: Callable
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    load_rule_config: Callable[..., dict]
    save_rule_config: Callable[[dict], None]


def create_fatigue_compliance_blueprint(
    dependencies: FatigueComplianceDependencies,
) -> Blueprint:
    """Create legacy-named compliance routes without changing their URLs."""

    blueprint = Blueprint("fatigue_compliance", __name__)

    @login_required
    def compliance_centre():
        """Retired standalone view; roster cells remain the monitoring surface."""
        if not dependencies.is_admin_user(current_user):
            abort(403)
        year, month = compliance_month(request.args.get("ym"))
        return redirect(url_for("roster_month", ym=f"{year:04d}-{month:02d}"))

    @login_required
    def admin_fatigue_rules():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        config = dependencies.load_rule_config()
        if request.method == "POST":
            dependencies.validate_csrf()
            action = request.form.get("action") or ""
            try:
                if action == "update_definitions":
                    definitions = {}
                    for key in (
                        "early_start_before",
                        "night_period_start",
                        "night_period_end",
                    ):
                        value = (request.form.get(key) or "").strip()
                        datetime.strptime(value, "%H:%M")
                        definitions[key] = value
                    if (
                        definitions["night_period_start"]
                        == definitions["night_period_end"]
                    ):
                        raise ValueError(
                            "The night-duty period must have a duration."
                        )
                    config["definitions"] = definitions
                    dependencies.save_rule_config(config)
                    flash(
                        "Duty time definitions updated for this airport.", "ok"
                    )
                elif action == "update_system":
                    code = (request.form.get("code") or "").upper()
                    if code not in config["system"]:
                        abort(404)
                    item = config["system"][code]
                    item["name"] = (
                        request.form.get("name") or item["name"]
                    ).strip()[:120]
                    item["severity"] = (
                        request.form.get("severity")
                        if request.form.get("severity")
                        in {"warning", "critical"}
                        else item["severity"]
                    )
                    item["enabled"] = request.form.get("enabled") == "on"
                    for key, parameter_item in item["parameters"].items():
                        value = float(request.form.get(
                            f"parameter_{key}", parameter_item["value"]
                        ))
                        if not 0 < value <= 10000:
                            raise ValueError(
                                f"{parameter_item['label']} must be greater "
                                "than zero."
                            )
                        parameter_item["value"] = value
                    dependencies.save_rule_config(config)
                    flash(f"{code} fatigue rule updated.", "ok")
                elif action == "add_custom":
                    rule_type = request.form.get("rule_type") or ""
                    if rule_type not in CUSTOM_FATIGUE_RULE_TYPES:
                        raise ValueError("Choose a supported rule check.")
                    name = (request.form.get("name") or "").strip()
                    if len(name) < 3:
                        raise ValueError("Give the rule a clear name.")
                    threshold = float(request.form.get("threshold") or 0)
                    if threshold <= 0:
                        raise ValueError("The limit must be greater than zero.")
                    type_meta = CUSTOM_FATIGUE_RULE_TYPES[rule_type]
                    window_days = int(request.form.get("window_days") or type_meta.get("default_window", 1))
                    if not 1 <= window_days <= 365:
                        raise ValueError("The review period must be 1–365 days.")
                    existing_codes = {str(item.get("code") or "") for item in config["custom"]}
                    sequence = 1
                    while f"CUSTOM_{sequence}" in existing_codes:
                        sequence += 1
                    config["custom"].append({
                        "code": f"CUSTOM_{sequence}", "name": name[:120],
                        "rule_type": rule_type, "threshold": threshold,
                        "window_days": window_days,
                        "severity": request.form.get("severity") if request.form.get("severity") in {"warning", "critical"} else "warning",
                        "enabled": request.form.get("enabled") == "on",
                    })
                    dependencies.save_rule_config(config)
                    flash("Custom fatigue rule saved.", "ok")
                elif action == "update_custom":
                    rule_type = request.form.get("rule_type") or ""
                    if rule_type not in CUSTOM_FATIGUE_RULE_TYPES:
                        raise ValueError("Choose a supported rule check.")
                    name = (request.form.get("name") or "").strip()
                    if len(name) < 3:
                        raise ValueError("Give the rule a clear name.")
                    threshold = float(request.form.get("threshold") or 0)
                    if threshold <= 0:
                        raise ValueError("The limit must be greater than zero.")
                    type_meta = CUSTOM_FATIGUE_RULE_TYPES[rule_type]
                    window_days = int(
                        request.form.get("window_days")
                        or type_meta.get("default_window", 1)
                    )
                    if not 1 <= window_days <= 365:
                        raise ValueError("The review period must be 1–365 days.")
                    severity = request.form.get("severity")
                    if severity not in {"warning", "critical"}:
                        severity = "warning"
                    code = (request.form.get("code") or "").upper()
                    existing = next((
                        item for item in config["custom"]
                        if item.get("code") == code
                    ), None)
                    if not existing:
                        abort(404)
                    existing.update({
                        "name": name[:120],
                        "rule_type": rule_type,
                        "threshold": threshold,
                        "window_days": window_days,
                        "severity": severity,
                        "enabled": request.form.get("enabled") == "on",
                    })
                    dependencies.save_rule_config(config)
                    flash(f"{existing['code']} fatigue rule saved.", "ok")
                elif action == "delete_custom":
                    code = (request.form.get("code") or "").upper()
                    before = len(config["custom"])
                    config["custom"] = [
                        item for item in config["custom"]
                        if item.get("code") != code
                    ]
                    if len(config["custom"]) == before:
                        abort(404)
                    dependencies.save_rule_config(config)
                    flash(f"{code} custom fatigue rule removed.", "ok")
                else:
                    abort(400)
            except (TypeError, ValueError) as exc:
                flash(str(exc), "error")
            return redirect(url_for("admin_fatigue_rules"))
        return render_template(
            "admin_fatigue_rules.html",
            system_rules=list(config["system"].values()),
            custom_rules=config["custom"],
            definitions=config["definitions"],
            rule_types=CUSTOM_FATIGUE_RULE_TYPES,
            current_unit=dependencies.db.session.get(
                dependencies.Unit, dependencies.current_unit_id()
            ),
        )

    @login_required
    def compliance_centre_export():
        """Retired with the standalone Compliance Centre."""
        if not dependencies.is_admin_user(current_user):
            abort(403)
        year, month = compliance_month(request.args.get("ym"))
        return redirect(url_for("roster_month", ym=f"{year:04d}-{month:02d}"))

    @blueprint.record_once
    def register_routes(state):
        routes = (
            ("/compliance-centre", "compliance_centre", compliance_centre, ["GET"]),
            (
                "/admin/fatigue-rules",
                "admin_fatigue_rules",
                admin_fatigue_rules,
                ["GET", "POST"],
            ),
            (
                "/compliance-centre/export",
                "compliance_centre_export",
                compliance_centre_export,
                ["GET"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
