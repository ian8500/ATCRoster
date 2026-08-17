"""POST action dispatch for roster administration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from flask import abort, flash, redirect, url_for


@dataclass(frozen=True)
class AdminActionDependencies:
    db: Any
    Watch: Any
    Staff: Any
    WorkPattern: Any
    StaffWatchHistory: Any
    StaffPatternAssignment: Any
    QualificationType: Any
    PersonQualification: Any
    ShiftType: Any
    Requirement: Any
    SpecialRequirement: Any
    RosterImpactEventType: Any
    JoinerDependencies: Any
    WatchConfigurationDependencies: Any
    ShiftConfigurationDependencies: Any
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    update_absence_types: Callable[..., tuple[str, str]]
    get_absence_types: Callable[..., list[dict[str, Any]]]
    save_absence_types: Callable[[Any], None]
    save_sms_settings: Callable[..., str | None]
    parse_sms_number_lines: Callable[[str], tuple[list[Any], list[str]]]
    save_roster_setting: Callable[[str, str], None]
    update_unit_roster_setup: Callable[..., tuple[str, str]]
    validate_pattern: Callable[[str | None], list[str]]
    parse_date: Callable[[str | None], Any]
    record_roster_impact: Callable[..., None]
    update_watch_configuration: Callable[..., tuple[str, str]]
    save_counter_mapping: Callable[..., None]
    create_joiner: Callable[..., Any]
    work_pattern_service: Any
    record_qualification_history: Callable[..., None]
    sync_qualification: Callable[..., None]
    now: Callable[[], Any]
    update_shift_definition: Callable[..., tuple[str, str]]
    parse_hhmm: Callable[[str | None], Any]
    prune_roster_code_settings: Callable[[int], int]
    refresh_shift_cache: Callable[[], None]
    clear_shift_groups_cache: Callable[[], None]
    save_monthly_requirements: Callable[..., None]
    save_special_requirement: Callable[..., str]
    delete_special_requirement: Callable[..., None]
    seed_toil_balances: Callable[..., tuple[int, int]]


def create_admin_action_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> AdminActionDependencies:
    """Bind administration actions to the canonical model registries."""
    return AdminActionDependencies(
        db=db,
        Watch=operational_models.Watch,
        Staff=operational_models.Staff,
        WorkPattern=saas_models.WorkPattern,
        StaffWatchHistory=operational_models.StaffWatchHistory,
        StaffPatternAssignment=saas_models.StaffPatternAssignment,
        QualificationType=saas_models.QualificationType,
        PersonQualification=saas_models.PersonQualification,
        ShiftType=operational_models.ShiftType,
        Requirement=operational_models.Requirement,
        SpecialRequirement=operational_models.SpecialRequirement,
        **services,
    )


def dispatch_admin_action(
    form: str, values: Mapping[str, str], dependencies: AdminActionDependencies
):
    """Dispatch one roster-administration form action."""
    deps = dependencies
    if form in {"absence_type_add", "absence_type_delete"}:
        deps.validate_csrf()
        message, category = deps.update_absence_types(
            form,
            values,
            load=deps.get_absence_types,
            save=deps.save_absence_types,
        )
        flash(message, category)
        return redirect(url_for("admin") + "#leave-types")

    if form == "sms_settings":
        deps.validate_csrf()
        error = deps.save_sms_settings(
            values,
            db=deps.db,
            parse_number_lines=deps.parse_sms_number_lines,
            save_setting=deps.save_roster_setting,
        )
        flash(
            error or "SMS numbers saved for this airport.",
            "error" if error else "ok",
        )
        return redirect(url_for("admin") + "#sms")

    if form == "unit_roster_setup":
        message, category = deps.update_unit_roster_setup(
            values,
            db=deps.db,
            validate_pattern=deps.validate_pattern,
            parse_date=deps.parse_date,
            save_setting=deps.save_roster_setting,
            record_roster_impact=deps.record_roster_impact,
            impact_type=deps.RosterImpactEventType.WATCH_PATTERN_CHANGE,
        )
        flash(message, category)
        return redirect(url_for("admin") + "#roster-setup")

    if form in {"watch_new", "watch_edit", "watch_delete"}:
        message, category = deps.update_watch_configuration(
            form,
            values,
            deps.WatchConfigurationDependencies(
                db=deps.db,
                Watch=deps.Watch,
                Staff=deps.Staff,
                StaffWatchHistory=deps.StaffWatchHistory,
                RosterImpactEventType=deps.RosterImpactEventType,
                current_unit_id=deps.current_unit_id,
                validate_pattern=deps.validate_pattern,
                parse_date=deps.parse_date,
                record_roster_impact=deps.record_roster_impact,
            ),
        )
        flash(message, category)
        return redirect(url_for("admin") + "#roster-setup")

    if form == "counter_mapping":
        try:
            deps.save_counter_mapping(
                values,
                db=deps.db,
                ShiftType=deps.ShiftType,
                unit_id=deps.current_unit_id(),
                save_setting=deps.save_roster_setting,
            )
        except ValueError as exc:
            abort(400, str(exc))
        flash("Shift counter mapping saved.", "ok")
        return redirect(url_for("admin") + "#shifts")

    if form == "staff_new":
        response = deps.create_joiner(
            values,
            deps.JoinerDependencies(
                db=deps.db,
                Staff=deps.Staff,
                WorkPattern=deps.WorkPattern,
                StaffWatchHistory=deps.StaffWatchHistory,
                StaffPatternAssignment=deps.StaffPatternAssignment,
                QualificationType=deps.QualificationType,
                PersonQualification=deps.PersonQualification,
                RosterImpactEventType=deps.RosterImpactEventType,
                current_unit_id=deps.current_unit_id,
                parse_date=deps.parse_date,
                work_pattern_service=deps.work_pattern_service,
                record_qualification_history=deps.record_qualification_history,
                sync_qualification=deps.sync_qualification,
                record_roster_impact=deps.record_roster_impact,
                now=deps.now,
            ),
        )
        if response is not None:
            return response

    if form in {"shift_new", "shift_edit", "shift_delete"}:
        message, category = deps.update_shift_definition(
            form,
            values,
            deps.ShiftConfigurationDependencies(
                db=deps.db,
                ShiftType=deps.ShiftType,
                QualificationType=deps.QualificationType,
                RosterImpactEventType=deps.RosterImpactEventType,
                current_unit_id=deps.current_unit_id,
                parse_hhmm=deps.parse_hhmm,
                record_roster_impact=deps.record_roster_impact,
                prune_roster_code_settings=deps.prune_roster_code_settings,
                refresh_shift_cache=deps.refresh_shift_cache,
                clear_shift_groups_cache=deps.clear_shift_groups_cache,
            ),
        )
        flash(message, category)
        return redirect(url_for("admin") + "#shifts")

    if form == "req":
        try:
            deps.save_monthly_requirements(
                values,
                db=deps.db,
                Requirement=deps.Requirement,
                unit_id=deps.current_unit_id(),
                impact_type=deps.RosterImpactEventType.STAFFING_REQUIREMENT_CHANGE,
                record_roster_impact=deps.record_roster_impact,
            )
        except ValueError as exc:
            abort(400, str(exc))
        flash("Requirements saved.", "ok")
        return redirect(url_for("admin") + "#requirements")

    if form == "special_requirement":
        try:
            message = deps.save_special_requirement(
                values,
                db=deps.db,
                SpecialRequirement=deps.SpecialRequirement,
                unit_id=deps.current_unit_id(),
                impact_type=deps.RosterImpactEventType.STAFFING_REQUIREMENT_CHANGE,
                record_roster_impact=deps.record_roster_impact,
            )
        except ValueError as exc:
            flash(str(exc), "error")
        else:
            flash(message, "ok")
        return redirect(url_for("admin") + "#requirements")

    if form == "special_requirement_delete":
        deps.delete_special_requirement(
            int(values.get("special_requirement_id") or 0),
            db=deps.db,
            SpecialRequirement=deps.SpecialRequirement,
            unit_id=deps.current_unit_id(),
            impact_type=deps.RosterImpactEventType.STAFFING_REQUIREMENT_CHANGE,
            record_roster_impact=deps.record_roster_impact,
        )
        flash("Special requirement removed.", "ok")
        return redirect(url_for("admin") + "#requirements")

    if form == "toil_seed":
        updated, errors = deps.seed_toil_balances(
            values.get("toil_seed_lines") or "", db=deps.db, Staff=deps.Staff
        )
        flash(
            f"TOIL balances updated: {updated} staff; {errors} error(s).",
            "ok" if errors == 0 else "error",
        )
        return redirect(url_for("admin"))
    return None
