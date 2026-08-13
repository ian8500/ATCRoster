"""Administration landing and policy-bound routes."""

from .blueprint import AdministrationDependencies, create_administration_blueprint

__all__ = ("AdministrationDependencies", "create_administration_blueprint")
