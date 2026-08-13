"""Application navigation and template-shell context."""

from .context import (
    NavigationContextDependencies,
    build_navigation_context,
    register_navigation_context,
)

__all__ = (
    "NavigationContextDependencies",
    "build_navigation_context",
    "register_navigation_context",
)
