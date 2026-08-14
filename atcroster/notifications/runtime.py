"""Bound notification delivery services for application composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .email import (
    email_service_configured,
    send_account_email,
    valid_email,
)
from .sms import (
    messagemedia_credentials,
    normalise_sms_number,
    normalise_uk_mobile,
)


@dataclass(frozen=True)
class NotificationRuntimeDependencies:
    db: Any
    app_logger: Any
    support_emails: Callable[[], list[str]]
    admin_emails: Callable[[int], list[str]]
    sms_configuration: Any
    sms_audit: Any
    overtime_sms: Any
    flash: Callable[[str, str], None]
    sms_sender: Callable[..., tuple[bool, str]]


class NotificationRuntime:
    """Expose configured notification operations through one domain boundary."""

    def __init__(self, dependencies: NotificationRuntimeDependencies) -> None:
        self.dependencies = dependencies

    credentials = staticmethod(messagemedia_credentials)
    normalise_number = staticmethod(normalise_sms_number)
    normalise_uk_mobile = staticmethod(normalise_uk_mobile)
    valid_email = staticmethod(valid_email)
    email_configured = staticmethod(email_service_configured)
    default_overtime_body: Any

    def number_options(self, key: str, unit_id: int | None = None) -> Any:
        return self.dependencies.sms_configuration.number_options(key, unit_id)

    def sender_options(self, unit_id: int | None = None) -> Any:
        return self.dependencies.sms_configuration.sender_options(unit_id)

    def operational_options(self, unit_id: int | None = None) -> Any:
        return self.dependencies.sms_configuration.operational_options(unit_id)

    def default_number(self, setting_key: str, options: Any, unit_id: int | None = None) -> str:
        return self.dependencies.sms_configuration.default_number(setting_key, options, unit_id)

    def sms_configured(self) -> bool:
        return self.dependencies.sms_configuration.service_configured()

    def sms_configuration_gaps(self) -> list[str]:
        return self.dependencies.sms_configuration.configuration_gaps()

    def send_email(self, to_address: str, subject: str, body: str) -> bool:
        return send_account_email(to_address, subject, body, self.dependencies.app_logger)

    def support_emails(self) -> list[str]:
        return self.dependencies.support_emails()

    def admin_emails(self, unit_id: int) -> list[str]:
        return self.dependencies.admin_emails(unit_id)

    def send_sms(self, *args: Any, **kwargs: Any) -> tuple[bool, str]:
        return self.dependencies.sms_sender(*args, **kwargs)

    def record_sms(self, **values: Any) -> None:
        self.dependencies.sms_audit.record(**values)

    def send_overtime(self, staff: list[Any], message: str) -> Any:
        return self.dependencies.overtime_sms.notify(staff, message)

    def flash_result(self, sent: int, failures: list[tuple[Any | None, str]]) -> None:
        if sent:
            suffix = "s" if sent != 1 else ""
            self.dependencies.flash(f"SMS sent to {sent} recipient{suffix}.", "ok")
        if failures:
            details = "; ".join(
                f"{staff.name if staff else 'System'}: {reason}"
                for staff, reason in failures[:8]
            )
            if len(failures) > 8:
                details += f"; and {len(failures) - 8} more"
            self.dependencies.flash(f"Some messages were not sent. {details}", "error")
