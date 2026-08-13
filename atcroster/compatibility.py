"""Narrow compatibility adapters for legacy module-level integrations."""

from __future__ import annotations

import sys
from typing import Any, Callable


def module_callback(module_name: str, attribute: str) -> Callable[..., Any]:
    """Resolve a legacy module hook when it is invoked.

    Some external integrations and tests replace public ``app`` module hooks
    after startup.  Keep that compatibility in one named adapter rather than
    spreading untyped ``globals()`` lookups through composition code.
    """

    def callback(*args: Any, **kwargs: Any) -> Any:
        return getattr(sys.modules[module_name], attribute)(*args, **kwargs)

    return callback
