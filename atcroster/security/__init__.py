"""Security boundaries registered by the application bootstrap."""

from .principal_boundaries import (
    PrincipalBoundaryDependencies,
    enforce_principal_boundaries,
    register_principal_boundaries,
)
from .decorators import create_admin_required, create_roster_edit_required

__all__ = (
    "PrincipalBoundaryDependencies",
    "enforce_principal_boundaries",
    "register_principal_boundaries",
    "create_admin_required",
    "create_roster_edit_required",
)
