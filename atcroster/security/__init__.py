"""Security boundaries registered by the application bootstrap."""

from .principal_boundaries import (
    PrincipalBoundaryDependencies,
    enforce_principal_boundaries,
    register_principal_boundaries,
)

__all__ = (
    "PrincipalBoundaryDependencies",
    "enforce_principal_boundaries",
    "register_principal_boundaries",
)
