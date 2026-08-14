"""Platform control-plane administration route."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, AbstractSet

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


@dataclass(frozen=True)
class PlatformAdminDependencies:
    db: Any
    PlatformIdentity: Any
    Unit: Any
    DatabaseRoutingMetadata: Any
    PlanHistory: Any
    SuperAdminAudit: Any
    UnitMembership: Any
    SecureInvitation: Any
    ProvisioningJob: Any
    SignupWorkflow: Any
    FeatureFlag: Any
    AggregateUsageEvent: Any
    now: Callable[[], Any]
    validate_csrf: Callable[[], None]
    consume_rate_limit: Callable[..., bool]
    security_event: Callable[..., None]
    feature_flags: AbstractSet[str]
    module_feature_flags: AbstractSet[str]
    metrics: Any
    worker_health_snapshot: Callable[..., dict[str, Any]]
    application_module: Any
    redis_health_check: Callable[[], None]
    invalidate_roster_cache: Callable[[int], None]
    deployment_environment: str


def create_platform_admin_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> PlatformAdminDependencies:
    """Bind control-plane records at the platform administration boundary."""
    return PlatformAdminDependencies(
        db=db,
        PlatformIdentity=saas_models.PlatformIdentity,
        Unit=operational_models.Unit,
        DatabaseRoutingMetadata=saas_models.DatabaseRoutingMetadata,
        PlanHistory=saas_models.PlanHistory,
        SuperAdminAudit=saas_models.SuperAdminAudit,
        UnitMembership=saas_models.UnitMembership,
        SecureInvitation=saas_models.SecureInvitation,
        ProvisioningJob=saas_models.ProvisioningJob,
        SignupWorkflow=saas_models.SignupWorkflow,
        FeatureFlag=saas_models.FeatureFlag,
        AggregateUsageEvent=saas_models.AggregateUsageEvent,
        **services,
    )


def create_platform_admin_blueprint(
    dependencies: PlatformAdminDependencies,
) -> Blueprint:
    """Build the SuperAdmin-only control-plane route."""
    blueprint = Blueprint("platform_administration", __name__)
    db = dependencies.db
    PlatformIdentity = dependencies.PlatformIdentity
    Unit = dependencies.Unit
    DatabaseRoutingMetadata = dependencies.DatabaseRoutingMetadata
    PlanHistory = dependencies.PlanHistory
    SuperAdminAudit = dependencies.SuperAdminAudit
    UnitMembership = dependencies.UnitMembership
    SecureInvitation = dependencies.SecureInvitation
    ProvisioningJob = dependencies.ProvisioningJob
    SignupWorkflow = dependencies.SignupWorkflow
    FeatureFlag = dependencies.FeatureFlag
    AggregateUsageEvent = dependencies.AggregateUsageEvent
    utcnow = dependencies.now
    _validate_csrf = dependencies.validate_csrf
    _consume_rate_limit = dependencies.consume_rate_limit
    _security_event = dependencies.security_event
    PLATFORM_FEATURE_FLAGS = dependencies.feature_flags
    PLATFORM_MODULE_FLAGS = dependencies.module_feature_flags

    def _serviceability(rows: list[dict[str, Any]]) -> dict[str, Any]:
        checks: dict[str, dict[str, str]] = {}
        try:
            db.session.execute(text("SELECT 1"))
            checks["database"] = {
                "status": "ready",
                "detail": "Control database reachable",
            }
        except Exception:
            db.session.rollback()
            checks["database"] = {
                "status": "blocking",
                "detail": "Control database unavailable",
            }
        try:
            dependencies.redis_health_check()
            checks["redis"] = {"status": "ready", "detail": "Shared Redis reachable"}
        except Exception:
            checks["redis"] = {
                "status": "blocking",
                "detail": "Shared Redis unavailable",
            }
        try:
            worker = dependencies.worker_health_snapshot(
                dependencies.application_module,
                stale_after_seconds=int(
                    os.environ.get("ATCROSTER_PROVISIONING_LEASE_SECONDS", "120")
                )
                * 2,
            )
        except Exception:
            worker = {
                "status": "unavailable",
                "active_workers": 0,
                "queue_depth": 0,
                "stale_workers": 0,
                "oldest_queued_age_seconds": 0,
            }
        worker_ready = worker.get("status") == "ready"
        checks["worker"] = {
            "status": "ready" if worker_ready else "warning",
            "detail": (
                f"{worker.get('active_workers', 0)} active; "
                f"{worker.get('queue_depth', 0)} queued"
                if worker_ready
                else "Provisioning worker unavailable"
            ),
        }
        unhealthy_routes = [
            row
            for row in rows
            if row["database_health"].lower() not in {"healthy", "ready", "active"}
            or row["provisioning_error"]
        ]
        checks["routing"] = {
            "status": "ready" if not unhealthy_routes else "warning",
            "detail": (
                f"{len(rows)} airport database route(s) healthy"
                if not unhealthy_routes
                else f"{len(unhealthy_routes)} airport route(s) require attention"
            ),
        }
        return {
            "checks": checks,
            "worker": worker,
            "unhealthy_routes": unhealthy_routes,
        }

    def _performance_snapshot() -> dict[str, Any]:
        routes: dict[str, dict[str, float]] = {}
        active_requests = 0.0
        for metric in dependencies.metrics.snapshot():
            name = metric["name"]
            labels = metric["labels"]
            value = float(metric["value"])
            if name == "active_web_requests":
                active_requests += value
                continue
            route = labels.get("route")
            if not route:
                continue
            row = routes.setdefault(route, {"requests": 0, "errors": 0, "seconds": 0})
            if name == "http_requests_total":
                row["requests"] += value
                if int(labels.get("status", "0")) >= 500:
                    row["errors"] += value
            elif name == "http_request_duration_seconds_sum":
                row["seconds"] += value
        output = []
        for route, values in routes.items():
            requests = int(values["requests"])
            output.append(
                {
                    "route": route,
                    "requests": requests,
                    "errors": int(values["errors"]),
                    "average_ms": round(values["seconds"] * 1000 / requests, 1)
                    if requests
                    else 0,
                }
            )
        output.sort(key=lambda row: (row["errors"], row["average_ms"]), reverse=True)
        total_requests = sum(row["requests"] for row in output)
        total_errors = sum(row["errors"] for row in output)
        return {
            "routes": output[:10],
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": round(total_errors * 100 / total_requests, 2)
            if total_requests
            else 0,
            "active_requests": max(0, int(active_requests)),
        }

    @login_required
    def platform_admin():
        """Privacy-preserving control plane: aggregates and unit metadata only."""
        if getattr(current_user, "role", "") != "superadmin":
            abort(403)
        platform_actor = PlatformIdentity.query.filter_by(
            username=current_user.username
        ).first()
        if not platform_actor:
            abort(403, "Super Admin identity is not provisioned in the control plane.")
        if request.method == "POST":
            _validate_csrf()
            action = (request.form.get("action") or "").strip()
            if action == "refresh_serviceability":
                db.session.add(
                    SuperAdminAudit(
                        actor_identity_id=platform_actor.id,
                        action="serviceability_rechecked",
                        safe_summary="Operator requested a fresh serviceability check",
                    )
                )
                db.session.commit()
                flash("Serviceability checks refreshed.", "success")
                return redirect(url_for("platform_admin"))
            elif action == "invalidate_roster_cache":
                unit = db.session.get(Unit, request.form.get("unit_id", type=int))
                if not unit or unit.status == "platform_control":
                    abort(404)
                dependencies.invalidate_roster_cache(unit.id)
                db.session.add(
                    SuperAdminAudit(
                        actor_identity_id=platform_actor.id,
                        unit_id=unit.id,
                        action="roster_cache_invalidated",
                        safe_summary=f"Invalidated roster cache for {unit.code}",
                    )
                )
                db.session.commit()
                flash(f"Roster cache cleared for {unit.code}.", "success")
                return redirect(url_for("platform_admin"))
            elif action == "create_unit":
                code = (request.form.get("code") or "").strip().upper()
                name = (request.form.get("name") or "").strip()
                plan = (request.form.get("plan") or "starter").strip()[:40]
                try:
                    limit = int(request.form.get("active_user_limit") or 10)
                except ValueError:
                    limit = 0
                if not re.fullmatch(r"[A-Z0-9]{2,12}", code):
                    flash("Airport code must be 2–12 letters or numbers.", "error")
                elif not name:
                    flash("Airport name is required.", "error")
                elif not 1 <= limit <= 10000:
                    flash("Active-user limit must be between 1 and 10,000.", "error")
                elif Unit.query.filter_by(code=code).first():
                    flash("That airport code already exists.", "error")
                else:
                    try:
                        unit = Unit(
                            code=code,
                            name=name,
                            plan=plan,
                            active_user_limit=limit,
                            onboarding_step=0,
                            status="provisioning",
                        )
                        db.session.add(unit)
                        db.session.flush()
                        db.session.add(
                            DatabaseRoutingMetadata(
                                unit_id=unit.id,
                                secret_name=f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL",
                                provisioning_state="pending",
                            )
                        )
                        db.session.add(
                            PlanHistory(
                                unit_id=unit.id,
                                plan=plan,
                                active_user_limit=limit,
                            )
                        )
                        db.session.add(
                            SuperAdminAudit(
                                actor_identity_id=platform_actor.id,
                                unit_id=unit.id,
                                action="airport_created",
                                safe_summary=f"Created airport {code} on {plan} plan with limit {limit}",
                            )
                        )
                        db.session.commit()
                        flash(
                            f"{name} metadata created. Configure its database "
                            "secret, then run provisioning.",
                            "ok",
                        )
                        return redirect(url_for("platform_admin"))
                    except Exception:
                        db.session.rollback()
                        raise
            elif action == "provision_unit":
                if not _consume_rate_limit(
                    "airport-provisioning",
                    platform_actor.id,
                    limit=10,
                    window=timedelta(hours=1),
                ):
                    abort(429, "Too many provisioning requests.")
                unit_id = int(request.form.get("unit_id") or 0)
                unit = db.session.get(Unit, unit_id)
                routing = db.session.get(DatabaseRoutingMetadata, unit_id)
                if not unit or unit.status == "platform_control" or not routing:
                    abort(404)
                active_accounts = UnitMembership.query.filter_by(
                    unit_id=unit_id, status="active"
                ).count()
                if active_accounts:
                    flash(
                        "This airport already has active accounts. Manage access "
                        "from the airport's Accounts page; bootstrap provisioning "
                        "is only for a new airport.",
                        "error",
                    )
                    return redirect(url_for("platform_admin"))
                existing_invitation = SecureInvitation.query.filter_by(
                    unit_id=unit_id,
                    role="UnitAdmin",
                    active_bootstrap_key="active",
                ).first()
                if existing_invitation:
                    flash(
                        "A bootstrap invitation is already active. Show that "
                        "one-time link or use Revoke and replace.",
                        "error",
                    )
                    return redirect(url_for("platform_admin"))
                active = (
                    ProvisioningJob.query.filter(
                        ProvisioningJob.unit_id == unit_id,
                        ProvisioningJob.state.in_(("queued", "running", "retry_wait")),
                    )
                    .with_for_update()
                    .first()
                )
                if not active:
                    active = ProvisioningJob(
                        unit_id=unit_id,
                        idempotency_key=hashlib.sha256(
                            f"{unit_id}:{secrets.token_hex(16)}".encode()
                        ).hexdigest(),
                        state="queued",
                        active_key="active",
                        next_attempt_at=utcnow(),
                    )
                    db.session.add(active)
                elif active.state == "retry_wait":
                    active.state = "queued"
                    active.next_attempt_at = utcnow()
                routing.provisioning_state = "queued"
                try:
                    db.session.commit()
                except IntegrityError:
                    # A concurrent request won the database uniqueness race.
                    # Treat this request as an idempotent resume.
                    db.session.rollback()
                    active = ProvisioningJob.query.filter_by(
                        unit_id=unit_id, active_key="active"
                    ).first()
                    if not active:
                        raise
                flash(
                    "Provisioning was queued. The worker will migrate and check "
                    "the airport database before issuing an invitation.",
                    "ok",
                )
                return redirect(url_for("platform_admin"))
            elif action == "cancel_provisioning":
                job = (
                    ProvisioningJob.query.filter_by(
                        id=int(request.form.get("job_id") or 0)
                    )
                    .with_for_update()
                    .first_or_404()
                )
                job.cancel_requested = True
                job.updated_at = utcnow()
                db.session.commit()
                flash("Provisioning cancellation requested.", "ok")
                return redirect(url_for("platform_admin"))
            elif action == "reveal_bootstrap":
                job = ProvisioningJob.query.filter_by(
                    id=int(request.form.get("job_id") or 0), state="completed"
                ).first_or_404()
                from platform_provisioning import pop_one_time_token

                raw_token = pop_one_time_token(job.id, job.unit_id)
                if raw_token:
                    invite_url = url_for(
                        "accept_invitation", token=raw_token, _external=True
                    )
                    flash(
                        "Copy this bootstrap link now; it will not be shown "
                        f"again: {invite_url}",
                        "ok",
                    )
                else:
                    flash(
                        "The one-time link is no longer available. Revoke the "
                        "pending bootstrap and deliberately issue a replacement.",
                        "error",
                    )
                return redirect(url_for("platform_admin"))
            elif action == "replace_bootstrap":
                unit_id = int(request.form.get("unit_id") or 0)
                unit = db.session.get(Unit, unit_id)
                if not unit or unit.status == "platform_control":
                    abort(404)
                if UnitMembership.query.filter_by(
                    unit_id=unit_id, status="active"
                ).count():
                    flash(
                        "This airport already has active accounts, so a bootstrap "
                        "invitation cannot be replaced.",
                        "error",
                    )
                    return redirect(url_for("platform_admin"))
                invitation = (
                    SecureInvitation.query.filter_by(
                        unit_id=unit_id,
                        role="UnitAdmin",
                        active_bootstrap_key="active",
                    )
                    .with_for_update()
                    .first()
                )
                if invitation:
                    invitation.disabled_at = utcnow()
                    invitation.active_bootstrap_key = None
                active = ProvisioningJob.query.filter_by(
                    unit_id=unit_id, active_key="active"
                ).first()
                if active:
                    flash("Provisioning is already in progress.", "error")
                else:
                    db.session.add(
                        ProvisioningJob(
                            unit_id=unit_id,
                            idempotency_key=hashlib.sha256(
                                f"{unit_id}:{secrets.token_hex(16)}".encode()
                            ).hexdigest(),
                            state="queued",
                            active_key="active",
                            next_attempt_at=utcnow(),
                        )
                    )
                    routing = db.session.get(DatabaseRoutingMetadata, unit_id)
                    if routing:
                        routing.provisioning_state = "queued"
                    flash("Replacement bootstrap generation was queued.", "ok")
                db.session.commit()
                return redirect(url_for("platform_admin"))
            elif action == "update_limit":
                try:
                    unit_id = int(request.form.get("unit_id") or 0)
                    new_limit = int(request.form.get("active_user_limit") or 0)
                except ValueError:
                    abort(400)
                unit = db.session.get(Unit, unit_id)
                if not unit or unit.status == "platform_control":
                    abort(404)
                if not 1 <= new_limit <= 10000:
                    flash("Active-user limit must be between 1 and 10,000.", "error")
                else:
                    active_count = UnitMembership.query.filter_by(
                        unit_id=unit.id, status="active"
                    ).count()
                    if new_limit < active_count:
                        flash(
                            f"Limit cannot be below the {active_count} active accounts.",
                            "error",
                        )
                    else:
                        old_limit = unit.active_user_limit
                        unit.active_user_limit = new_limit
                        db.session.add(
                            PlanHistory(
                                unit_id=unit.id,
                                plan=unit.plan,
                                active_user_limit=new_limit,
                            )
                        )
                        db.session.add(
                            SuperAdminAudit(
                                actor_identity_id=platform_actor.id,
                                unit_id=unit.id,
                                action="account_limit_changed",
                                safe_summary=f"Changed active-user limit from {old_limit} to {new_limit}",
                            )
                        )
                        db.session.commit()
                        flash(f"{unit.code} account limit updated.", "ok")
                        return redirect(url_for("platform_admin"))
            elif action == "toggle_suspension":
                unit_id = int(request.form.get("unit_id") or 0)
                unit = db.session.get(Unit, unit_id)
                if not unit or unit.status == "platform_control":
                    abort(404)
                if unit.status == "suspended":
                    unit.status = "active"
                    unit.suspended_at = None
                    action_name = "airport_restored"
                else:
                    unit.status = "suspended"
                    unit.suspended_at = utcnow()
                    action_name = "airport_suspended"
                db.session.add(
                    SuperAdminAudit(
                        actor_identity_id=platform_actor.id,
                        unit_id=unit.id,
                        action=action_name,
                        safe_summary=f"{action_name}: {unit.code}",
                    )
                )
                db.session.commit()
                return redirect(url_for("platform_admin"))
            elif action == "delete_unit":
                unit_id = int(request.form.get("unit_id") or 0)
                confirmation = (
                    (request.form.get("confirmation_code") or "").strip().upper()
                )
                database_acknowledged = request.form.get("database_retained") == "yes"
                unit = db.session.get(Unit, unit_id)
                if not unit or unit.status == "platform_control":
                    abort(404)
                if confirmation != unit.code.upper():
                    flash(
                        f"Type {unit.code} exactly to confirm airport deletion.",
                        "error",
                    )
                    return redirect(url_for("platform_admin"))
                if not database_acknowledged:
                    flash(
                        "Confirm that the separate airport database will be "
                        "retained for deliberate backup and decommissioning.",
                        "error",
                    )
                    return redirect(url_for("platform_admin"))
                active_accounts = UnitMembership.query.filter_by(
                    unit_id=unit.id, status="active"
                ).count()
                active_job = ProvisioningJob.query.filter(
                    ProvisioningJob.unit_id == unit.id,
                    ProvisioningJob.state.in_(
                        (
                            "queued",
                            "running",
                            "retry_wait",
                        )
                    ),
                ).first()
                if active_accounts:
                    flash(
                        "Suspend or remove every active airport account before "
                        "deleting the airport.",
                        "error",
                    )
                    return redirect(url_for("platform_admin"))
                if active_job:
                    flash(
                        "Cancel and finish the active provisioning job before "
                        "deleting the airport.",
                        "error",
                    )
                    return redirect(url_for("platform_admin"))

                invitation_ids = [
                    row.id
                    for row in SecureInvitation.query.filter_by(unit_id=unit.id).all()
                ]
                membership_ids = [
                    row.id
                    for row in UnitMembership.query.filter_by(unit_id=unit.id).all()
                ]
                workflow_filters = []
                if invitation_ids:
                    workflow_filters.append(
                        SignupWorkflow.invitation_id.in_(invitation_ids)
                    )
                if membership_ids:
                    workflow_filters.append(
                        SignupWorkflow.membership_id.in_(membership_ids)
                    )
                if workflow_filters:
                    db.session.query(SignupWorkflow).filter(
                        db.or_(*workflow_filters)
                    ).delete(synchronize_session=False)

                job_ids = [
                    row.id
                    for row in ProvisioningJob.query.filter_by(unit_id=unit.id).all()
                ]
                db.session.query(SuperAdminAudit).filter_by(unit_id=unit.id).update(
                    {"unit_id": None}, synchronize_session=False
                )
                for model in (
                    SecureInvitation,
                    ProvisioningJob,
                    FeatureFlag,
                    PlanHistory,
                    AggregateUsageEvent,
                    DatabaseRoutingMetadata,
                    UnitMembership,
                ):
                    db.session.query(model).filter_by(unit_id=unit.id).delete(
                        synchronize_session=False
                    )
                deleted_code = unit.code
                db.session.delete(unit)
                db.session.add(
                    SuperAdminAudit(
                        actor_identity_id=platform_actor.id,
                        unit_id=None,
                        action="airport_deleted",
                        safe_summary=(
                            f"Deleted airport metadata for {deleted_code}; "
                            "operational database retained for decommissioning."
                        ),
                    )
                )
                db.session.commit()
                if job_ids and os.environ.get("REDIS_URL"):
                    try:
                        import redis

                        cache = redis.from_url(
                            os.environ["REDIS_URL"],
                            socket_connect_timeout=2,
                            socket_timeout=2,
                            decode_responses=True,
                        )
                        cache.delete(
                            *[
                                f"atcroster:provisioning-token:{job_id}"
                                for job_id in job_ids
                            ]
                        )
                    except Exception:
                        _security_event(
                            "airport_token_cleanup_failed",
                            unit_digest=hashlib.sha256(
                                str(unit_id).encode()
                            ).hexdigest()[:16],
                        )
                flash(
                    f"{deleted_code} airport metadata deleted. Its separate "
                    "database was retained and must be backed up or destroyed "
                    "through the database provider.",
                    "ok",
                )
                return redirect(url_for("platform_admin"))
            elif action == "set_feature":
                try:
                    unit_id = int(request.form.get("unit_id") or 0)
                except ValueError:
                    abort(400)
                key = (request.form.get("key") or "").strip()
                if key not in PLATFORM_FEATURE_FLAGS:
                    abort(400, "Unknown feature flag.")
                unit = db.session.get(Unit, unit_id)
                if not unit or unit.status == "platform_control":
                    abort(404)
                row = FeatureFlag.query.filter_by(unit_id=unit.id, key=key).first()
                if not row:
                    row = FeatureFlag(unit_id=unit.id, key=key)
                    db.session.add(row)
                old_enabled = bool(row.enabled)
                row.enabled = request.form.get("enabled") == "yes"
                db.session.add(
                    SuperAdminAudit(
                        actor_identity_id=platform_actor.id,
                        unit_id=unit.id,
                        action="feature_flag_changed",
                        safe_summary=(
                            f"Changed {key} from {old_enabled} to {row.enabled}"
                        ),
                    )
                )
                db.session.commit()
                return redirect(url_for("platform_admin"))
            else:
                abort(400)
        rows = []
        now = utcnow()
        for unit in (
            Unit.query.filter(Unit.status != "platform_control")
            .order_by(Unit.code)
            .all()
        ):
            active_accounts = UnitMembership.query.filter_by(
                unit_id=unit.id, status="active"
            ).count()
            flags = {
                row.key: row.enabled
                for row in FeatureFlag.query.filter_by(unit_id=unit.id).all()
            }
            if "competency_module" not in flags:
                flags["competency_module"] = bool(flags.get("training_module"))
            routing = db.session.get(DatabaseRoutingMetadata, unit.id)
            activity = (
                db.session.query(
                    db.func.coalesce(db.func.sum(AggregateUsageEvent.count), 0)
                )
                .filter(AggregateUsageEvent.unit_id == unit.id)
                .scalar()
            )
            bootstrap = (
                SecureInvitation.query.filter_by(unit_id=unit.id, role="UnitAdmin")
                .order_by(SecureInvitation.id.desc())
                .first()
            )
            latest_job = (
                ProvisioningJob.query.filter_by(unit_id=unit.id)
                .order_by(ProvisioningJob.id.desc())
                .first()
            )
            if (
                latest_job
                and latest_job.state == "completed"
                and latest_job.last_error_code == "bootstrap_already_issued"
            ):
                latest_job = (
                    ProvisioningJob.query.filter_by(
                        unit_id=unit.id,
                        state="completed",
                        last_error_code="",
                    )
                    .order_by(ProvisioningJob.id.desc())
                    .first()
                )
            if not bootstrap:
                bootstrap_status = "established" if active_accounts else "not issued"
            elif bootstrap.accepted_at:
                bootstrap_status = "accepted"
            elif bootstrap.disabled_at:
                bootstrap_status = "revoked"
            else:
                comparison_now = (
                    now.replace(tzinfo=None)
                    if bootstrap.expires_at.tzinfo is None
                    else now
                )
                bootstrap_status = (
                    "expired" if bootstrap.expires_at <= comparison_now else "unused"
                )
            rows.append(
                {
                    "unit": unit,
                    "active_accounts": active_accounts,
                    "flags": flags,
                    "database_health": routing.health if routing else "unknown",
                    "provisioning_state": (
                        routing.provisioning_state if routing else "pending"
                    ),
                    "provisioning_error": (routing.last_error_code if routing else ""),
                    "migration_version": routing.migration_version if routing else "",
                    "storage_bytes": routing.storage_bytes if routing else 0,
                    "activity_count": int(activity or 0),
                    "bootstrap_status": bootstrap_status,
                    "provisioning_job": latest_job,
                }
            )
        serviceability = _serviceability(rows)
        performance = _performance_snapshot()
        issues = []
        for component, check in serviceability["checks"].items():
            if check["status"] != "ready":
                issues.append(
                    {
                        "severity": check["status"],
                        "component": component.title(),
                        "summary": check["detail"],
                        "remediation": {
                            "redis": "Check the Redis service and connection settings.",
                            "worker": "Check the provisioning worker deployment and recent logs.",
                            "routing": "Review the affected airport database state below.",
                            "database": "Check the control database service and connection settings.",
                        }.get(component, "Review the service configuration and logs."),
                    }
                )
        if performance["total_errors"]:
            issues.append(
                {
                    "severity": "warning",
                    "component": "Web requests",
                    "summary": f"{performance['total_errors']} server error response(s) observed",
                    "remediation": "Review the affected routes and correlated deployment logs.",
                }
            )
        recent_actions = (
            SuperAdminAudit.query.order_by(SuperAdminAudit.occurred_at.desc())
            .limit(10)
            .all()
        )
        return render_template(
            "platform_admin.html",
            rows=rows,
            module_feature_keys=sorted(PLATFORM_MODULE_FLAGS),
            serviceability=serviceability,
            performance=performance,
            issues=issues,
            recent_actions=recent_actions,
            deployment_environment=dependencies.deployment_environment,
            deployment_version=(
                os.environ.get("ATCROSTER_COMMIT_SHA")
                or os.environ.get("RAILWAY_GIT_COMMIT_SHA")
                or "unreported"
            )[:12],
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/platform/admin", "platform_admin", platform_admin, methods=("GET", "POST")
        )

    return blueprint
