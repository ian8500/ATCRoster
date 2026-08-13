"""Access policy shared by training and competency views."""

from __future__ import annotations

from typing import Any, Callable


def may_view_training_profile(
    person: Any,
    actor: Any,
    *,
    is_editor: Callable[[Any], bool],
    can_manage_training: Callable[[Any], bool],
    can_record_training: Callable[[Any], bool],
) -> bool:
    """Allow a person to view their profile or an authorised trainer to do so."""
    return bool(
        person.id == actor.id
        or is_editor(actor)
        or can_manage_training(actor)
        or can_record_training(actor)
    )
