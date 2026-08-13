"""Safe post-authentication redirect policy."""

from __future__ import annotations

import re
from typing import Callable
from urllib import parse as urllib_parse


LOGIN_NEXT_ENDPOINTS = {
    "/": "index", "/modules": "module_home", "/briefing/": "briefing.home",
    "/training/": "training_home", "/competency/": "competency_home", "/admin": "admin",
    "/administration": "administration_home", "/administration/kiosk-accounts": "kiosk_accounts",
    "/leave": "leave", "/messages": "unit_messages", "/overtime": "overtime",
    "/platform/admin": "platform_admin", "/reports": "reports_index",
    "/reports/leave-year": "report_leave_year", "/reports/sickness": "report_sickness",
    "/requests": "requests_page", "/unit/accounts": "unit_accounts",
}


def canonical_login_redirect(target: str | None, *, url_for: Callable[..., str], default_endpoint: str = "index", user_id: int | None = None) -> str:
    """Return only URLs generated from explicit internal route allowlists."""
    if not target:
        return url_for(default_endpoint)
    parsed = urllib_parse.urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return url_for(default_endpoint)
    if endpoint := LOGIN_NEXT_ENDPOINTS.get(parsed.path):
        return url_for(endpoint)
    if match := re.fullmatch(r"/roster/(\d{4}-\d{2})", parsed.path):
        return url_for("roster_month", ym=match.group(1))
    if match := re.fullmatch(r"/reports/leave/(\d{4}-\d{2})", parsed.path):
        return url_for("report_leave", ym=match.group(1))
    if match := re.fullmatch(r"/staff/(\d+)", parsed.path):
        if user_id and int(match.group(1)) == user_id:
            return url_for("staff_profile", sid=user_id)
    return url_for(default_endpoint)


def airport_login_endpoint(user: object) -> str:
    """Select the module landing endpoint for an authenticated airport user."""
    if getattr(user, "role", "") == "position_monitor":
        return "live_position.kiosk_hmi"
    return "module_home"
