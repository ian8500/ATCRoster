"""Application navigation and template-shell context."""

from .context import (
    NavigationContextDependencies,
    build_navigation_context,
    create_navigation_context_dependencies,
    register_navigation_context,
)

__all__ = (
    "NavigationContextDependencies",
    "build_navigation_context",
    "create_navigation_context_dependencies",
    "register_navigation_context",
)
