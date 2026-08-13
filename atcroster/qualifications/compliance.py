"""Monthly operational fatigue-compliance reporting."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ComplianceRuntimeDependencies:
    Assignment: Any
    Staff: Any
    Watch: Any
    month_range: Callable[..., tuple[Any, list[Any]]]
    fatigue_rule_config: Callable[[], dict[str, Any]]
    fatigue_flags_for_range: Callable[[Any, list[Any]], dict[Any, list[str]]]


def create_compliance_runtime_dependencies(
    *, operational_models: Any, **services: Any
) -> ComplianceRuntimeDependencies:
    """Bind compliance reporting to operational roster records."""
    return ComplianceRuntimeDependencies(
        Assignment=operational_models.Assignment,
        Staff=operational_models.Staff,
        Watch=operational_models.Watch,
        **services,
    )


class ComplianceRuntime:
    """Own monthly qualification and fatigue-compliance report assembly."""

    def __init__(self, dependencies: ComplianceRuntimeDependencies) -> None:
        self.dependencies = dependencies

    def findings(self, year: int, month: int) -> dict[str, Any]:
        deps = self.dependencies
        return monthly_compliance_findings(
            year,
            month,
            Assignment=deps.Assignment,
            Staff=deps.Staff,
            Watch=deps.Watch,
            month_range=deps.month_range,
            fatigue_rule_config=deps.fatigue_rule_config,
            fatigue_flags_for_range=deps.fatigue_flags_for_range,
        )


def monthly_compliance_findings(
    year: int,
    month: int,
    *,
    Assignment: Any,
    Staff: Any,
    Watch: Any,
    month_range: Callable[..., tuple[Any, list[Any]]],
    fatigue_rule_config: Callable[[], dict[str, Any]],
    fatigue_flags_for_range: Callable[[Any, list[Any]], dict[Any, list[str]]],
) -> dict[str, Any]:
    """Build the staff-by-rule compliance report for a roster month."""
    _, days = month_range(year, month)
    people = (
        Staff.query.filter_by(is_operational=True)
        .outerjoin(Watch, Staff.watch_id == Watch.id)
        .order_by(Watch.order_index, Staff.name)
        .all()
    )
    rows = []
    rule_counts: Counter[str] = Counter()
    rule_config = fatigue_rule_config()
    metadata = dict(rule_config["system"])
    metadata.update({str(item.get("code")): item for item in rule_config["custom"]})
    for person in people:
        issues = []
        flags = fatigue_flags_for_range(person, days)
        for finding_day, messages in sorted(flags.items()):
            assignment = Assignment.query.filter_by(
                staff_id=person.id, day=finding_day
            ).first()
            for message in messages:
                match = re.search(r"\b(D\d{2}|USR-[A-F0-9]+)\b", message)
                code = match.group(1) if match else ""
                rule_metadata = metadata.get(code, {})
                rule = str(
                    rule_metadata.get("name")
                    or message.split(":", 1)[0].split("(", 1)[0].strip()
                )
                severity = rule_metadata.get("severity")
                if severity not in {"warning", "critical"}:
                    severity = (
                        "critical"
                        if any(
                            token in message
                            for token in (
                                "<11h",
                                "3rd consecutive",
                                ">200h",
                                "> 10h",
                            )
                        )
                        else "warning"
                    )
                rule_counts[f"{code} · {rule}" if code else rule] += 1
                issues.append(
                    {
                        "day": finding_day,
                        "message": message,
                        "rule": rule,
                        "rule_code": code,
                        "severity": severity,
                        "assignment": assignment,
                    }
                )
        rows.append({"staff": person, "issues": issues, "total": len(issues)})
    return {
        "days": days,
        "rows": rows,
        "total": sum(row["total"] for row in rows),
        "affected": sum(1 for row in rows if row["total"]),
        "critical": sum(
            1
            for row in rows
            for issue in row["issues"]
            if issue["severity"] == "critical"
        ),
        "rule_counts": rule_counts.most_common(),
    }
