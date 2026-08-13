"""Authenticated password-change route extracted from ``app.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class PasswordDependencies:
    db: Any
    Staff: Any
    PlatformIdentity: Any
    tenant_get: Callable[[Any, int], Any]
    validate_csrf: Callable[[], None]
    generate_password_hash: Callable[[str], str]


def create_password_blueprint(dependencies: PasswordDependencies) -> Blueprint:
    blueprint = Blueprint("account_password", __name__)

    def security_redirect():
        return redirect(
            url_for("password_change") if current_user.role == "superadmin"
            else url_for("staff_profile", sid=current_user.id) + "#security"
        )

    @login_required
    def password_change():
        if request.method == "POST":
            dependencies.validate_csrf()
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirmation = request.form.get("confirm_password", "")
            if not current_user.check_password(current_password):
                flash("Current password is incorrect.", "error")
                return security_redirect()
            if not new_password or new_password != confirmation:
                flash("New passwords do not match.", "error")
                return security_redirect()
            if len(new_password) < 12:
                flash("Use a password of at least 12 characters.", "error")
                return security_redirect()
            if new_password == current_password:
                flash("Choose a password different from the current password.", "error")
                return security_redirect()
            if current_user.role == "superadmin":
                identity = dependencies.PlatformIdentity.query.filter_by(username=current_user.username).first_or_404()
                identity.password_hash = dependencies.generate_password_hash(new_password)
            else:
                staff = dependencies.tenant_get(dependencies.Staff, current_user.id)
                if not staff:
                    abort(404)
                staff.set_password(new_password)
                identity = dependencies.PlatformIdentity.query.filter_by(username=staff.username).first()
                if identity:
                    identity.password_hash = staff.password_hash
            dependencies.db.session.commit()
            flash("Password updated.", "ok")
            return redirect(url_for("platform_admin") if current_user.role == "superadmin" else url_for("staff_profile", sid=current_user.id) + "#security")
        return render_template("password.html")

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule("/password", "password_change", password_change, methods=("GET", "POST"))

    return blueprint
