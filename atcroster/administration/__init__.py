"""Administration landing and policy-bound routes."""

from .blueprint import AdministrationDependencies, create_administration_blueprint
from .toil import ToilAdministrationDependencies, create_toil_administration_blueprint

__all__ = (
    "AdministrationDependencies",
    "ToilAdministrationDependencies",
    "create_administration_blueprint",
    "create_toil_administration_blueprint",
)
