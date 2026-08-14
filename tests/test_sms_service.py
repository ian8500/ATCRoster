from datetime import date

from atcroster.notifications.sms import (
    clicksend_credentials, normalise_sms_number, normalise_uk_mobile,
    parse_sms_number_lines, send_via_clicksend,
)
from atcroster.notifications.email import valid_email
from atcroster.notifications.configuration import SmsConfigurationService
from atcroster.notifications.audit import SmsAuditService
from atcroster.notifications.overtime import OvertimeSmsService, default_overtime_sms_body


def test_normalise_sms_number_accepts_e164_display_punctuation():
    assert normalise_sms_number("+44 (7700) 900-123") == "+447700900123"
    assert normalise_sms_number("07700900123") == ""
    assert normalise_sms_number("+123") == ""


def test_normalise_uk_mobile_accepts_supported_input_forms():
    assert normalise_uk_mobile("07700 900123") == "+447700900123"
    assert normalise_uk_mobile("00447700900123") == "+447700900123"
    assert normalise_uk_mobile("+441234567890") == ""


def test_valid_email_normalizes_and_rejects_invalid_addresses():
    assert valid_email("  ADMIN@EXAMPLE.COM ") == "admin@example.com"
    assert valid_email("not-an-email") == ""
    assert valid_email("admin@-example.com") == ""
    assert valid_email("admin@example..com") == ""


def test_parse_sms_number_lines_deduplicates_and_reports_invalid_lines():
    parsed, errors = parse_sms_number_lines("Ops | +44 7700 900123\n+447700900123\nbad")
    assert parsed == [{"number": "+447700900123", "label": "Ops"}]
    assert errors == ["line 3"]


def test_sms_configuration_uses_only_configured_unit_numbers(monkeypatch):
    monkeypatch.setenv("CLICK_SEND_DEFAULT_SENDER", "+447700900124")
    service = SmsConfigurationService(
        settings_snapshot=lambda unit_id: {
            "sms_sender_numbers": '[{"number": "+447700900123", "label": "Ops"}]',
            "sms_default_sender": "+447700900123",
        },
        current_unit_id=lambda: 7,
    )
    options = service.sender_options()
    assert options == [
        {"number": "+447700900123", "label": "Ops"},
        {"number": "+447700900124", "label": "Unit fallback sender"},
    ]
    assert service.default_number("sms_default_sender", options) == "+447700900123"


def test_clicksend_send_uses_documented_basic_auth_and_message_shape(monkeypatch):
    monkeypatch.setenv("CLICK_SEND_USERNAME", "clicksend-user")
    monkeypatch.setenv("CLICK_SEND_API_KEY", "clicksend-key")
    captured = {}

    class Response:
        status = 200

        def read(self, size=-1):
            assert size == 64 * 1024
            return b'{"response_code":"SUCCESS","data":{"messages":[{"message_id":"cs-1"}]}}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = request.data
        captured["authorization"] = request.get_header("Authorization")
        assert timeout == 10
        return Response()

    monkeypatch.setattr("atcroster.notifications.sms.urllib_request.urlopen", fake_urlopen)
    assert clicksend_credentials()[:2] == ("clicksend-user", "clicksend-key")
    assert send_via_clicksend("+447700900124", "Test", "+447700900123") == (True, "cs-1")
    assert captured["url"] == "https://rest.clicksend.com/v3/sms/send"
    assert b'"body": "Test"' in captured["payload"]
    assert captured["authorization"].startswith("Basic ")


def test_sms_audit_service_stamps_current_unit_and_actor():
    added = []

    class Session:
        def add(self, record):
            added.append(record)

        def commit(self):
            pass

    class Audit:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    service = SmsAuditService(
        db=type("Database", (), {"session": Session()})(), SmsAudit=Audit,
        current_unit_id=lambda: 4,
        current_user=lambda: type("User", (), {"id": 9, "name": "Manager"})(),
    )
    service.record(sender_number="+44 7700 900123", recipient_number="+447700900124",
                   recipient_label="Duty desk", body="Operational update",
                   message_type="operational", provider_message_id="provider-1")
    assert added[0].unit_id == 4
    assert added[0].sent_by_staff_id == 9
    assert added[0].sender_number == "+447700900123"


def test_overtime_sms_service_sends_and_audits_eligible_staff():
    records = []
    configuration = type("Configuration", (), {
        "sender_options": lambda self: [{"number": "+447700900123"}],
        "default_number": lambda self, key, options: options[0]["number"],
        "service_configured": lambda self: True,
    })()
    audit = type("Audit", (), {"record": lambda self, **kwargs: records.append(kwargs)})()
    service = OvertimeSmsService(configuration, audit, lambda *_args: (True, "provider-id"))
    staff = type("Staff", (), {"name": "Alex", "phone_number": "+447700900124"})()
    assert service.notify([staff], "Available") == (1, [])
    assert records[0]["message_type"] == "overtime"
    assert default_overtime_sms_body(date(2026, 8, 14), "M").startswith("Overtime available")
