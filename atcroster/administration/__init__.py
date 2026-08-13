"""Administration landing and policy-bound routes."""

from .blueprint import AdministrationDependencies, create_administration_blueprint
from .actions import AdminActionDependencies, dispatch_admin_action
from .toil import (
    ToilAdministrationDependencies,
    create_toil_administration_blueprint,
    seed_toil_balances,
)

__all__ = (
    "AdministrationDependencies",
    "AdminActionDependencies",
    "ToilAdministrationDependencies",
    "create_administration_blueprint",
    "dispatch_admin_action",
    "create_toil_administration_blueprint",
    "seed_toil_balances",
)
