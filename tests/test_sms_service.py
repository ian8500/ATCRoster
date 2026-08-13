from atcroster.notifications.sms import (
    normalise_sms_number, normalise_uk_mobile, parse_sms_number_lines,
)
from atcroster.notifications.email import valid_email
from atcroster.notifications.configuration import SmsConfigurationService


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
    monkeypatch.setenv("MESSAGEMEDIA_FALLBACK_SENDER", "+447700900124")
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
