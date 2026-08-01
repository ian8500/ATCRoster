from types import SimpleNamespace

from access_policy import (
    has_permission,
    is_admin,
    may_apply_annotations,
    may_edit_roster,
    may_manage_training,
    may_override_roster_conflicts,
    permissions_for,
)


def _user(**values):
    defaults = {
        "role": "viewer",
        "is_admin": False,
        "is_wm": False,
        "is_dwm": False,
        "has_assessor": False,
        "permissions_json": "{}",
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_admin_role_and_legacy_admin_flag_are_both_recognised():
    assert is_admin(_user(role="admin")) is True
    assert is_admin(_user(is_admin=True)) is True
    assert is_admin(_user(role="editor")) is False


def test_invalid_or_non_object_permissions_default_to_deny():
    assert permissions_for(_user(permissions_json="not-json")) == {}
    assert permissions_for(_user(permissions_json='["edit_roster"]')) == {}
    assert permissions_for(_user(permissions_json={"edit_roster": True})) == {}
    assert has_permission(_user(), "edit_roster") is False


def test_watch_manager_needs_explicit_permission_to_edit_roster():
    assert may_edit_roster(_user(is_wm=True)) is False
    assert (
        may_edit_roster(_user(is_wm=True, permissions_json='{"edit_roster": true}'))
        is True
    )


def test_annotation_and_conflict_permissions_are_independent():
    annotation_user = _user(permissions_json='{"apply_annotations": true}')
    override_user = _user(permissions_json='{"override_roster_conflicts": true}')
    assert may_apply_annotations(annotation_user) is True
    assert may_override_roster_conflicts(annotation_user) is False
    assert may_override_roster_conflicts(override_user) is True


def test_operational_manager_can_manage_training_without_admin_role():
    assert may_manage_training(_user(is_dwm=True)) is True
