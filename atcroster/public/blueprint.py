"""Public and legal pages with stable legacy endpoint names."""

from __future__ import annotations

import os

from flask import Blueprint, current_app, render_template, send_from_directory

public_blueprint = Blueprint("public", __name__)


def legal_context() -> dict[str, str]:
    """Return public contact details used by the legal and privacy notices."""
    return {
        "legal_entity": os.environ.get(
            "ATCROSTER_LEGAL_ENTITY",
            "Readback Correct",
        ).strip(),
        "privacy_email": os.environ.get(
            "ATCROSTER_PRIVACY_EMAIL",
            os.environ.get("ATCROSTER_SUPPORT_EMAIL", "privacy@atcroster.com"),
        ).strip(),
        "legal_address": os.environ.get(
            "ATCROSTER_LEGAL_ADDRESS",
            "Flat 0/2, 24 Caird Drive, Glasgow, Scotland, G11 5DT",
        ).strip(),
        "company_number": os.environ.get("ATCROSTER_COMPANY_NUMBER", "").strip(),
        "policy_date": "27 July 2026",
    }


def favicon():
    return send_from_directory(
        current_app.static_folder,
        "favicon.svg",
        mimetype="image/svg+xml",
    )


def privacy_notice():
    return render_template("privacy.html", **legal_context())


def cookie_notice():
    return render_template("cookies.html", **legal_context())


def terms_of_service():
    return render_template("terms.html", **legal_context())


def subprocessor_notice():
    return render_template("subprocessors.html", **legal_context())


@public_blueprint.record_once
def register_legacy_endpoints(state) -> None:
    """Register through the blueprint without changing public endpoint names.

    Flask prefixes ordinary blueprint endpoints. These long-lived endpoints are
    referenced by templates and are therefore installed explicitly on the app.
    """
    routes = (
        ("/favicon.ico", "favicon", favicon),
        ("/privacy", "privacy_notice", privacy_notice),
        ("/cookies", "cookie_notice", cookie_notice),
        ("/terms", "terms_of_service", terms_of_service),
        ("/subprocessors", "subprocessor_notice", subprocessor_notice),
    )
    for rule, endpoint, view_func in routes:
        state.app.add_url_rule(rule, endpoint, view_func, methods=("GET",))
