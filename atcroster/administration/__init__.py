"""Administration landing and policy-bound routes."""

from .blueprint import (
    AdminDashboardDependencies,
    AdministrationDependencies,
    create_admin_dashboard_blueprint,
    create_administration_blueprint,
)
from .actions import AdminActionDependencies, dispatch_admin_action
from .toil import (
    ToilService,
    ToilServiceDependencies,
    ToilAdministrationDependencies,
    create_toil_administration_blueprint,
    seed_toil_balances,
    annotation_accrual_half_days,
    apply_annotation_toil_delta,
    accrued_and_used_half_days,
)

__all__ = (
    "AdministrationDependencies",
    "AdminDashboardDependencies",
    "AdminActionDependencies",
    "ToilAdministrationDependencies",
    "ToilService",
    "ToilServiceDependencies",
    "create_administration_blueprint",
    "create_admin_dashboard_blueprint",
    "dispatch_admin_action",
    "create_toil_administration_blueprint",
    "seed_toil_balances",
    "annotation_accrual_half_days",
    "apply_annotation_toil_delta",
    "accrued_and_used_half_days",
)
