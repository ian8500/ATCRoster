from atcroster.notifications.sms import normalise_sms_number, normalise_uk_mobile


def test_normalise_sms_number_accepts_e164_display_punctuation():
    assert normalise_sms_number("+44 (7700) 900-123") == "+447700900123"
    assert normalise_sms_number("07700900123") == ""
    assert normalise_sms_number("+123") == ""


def test_normalise_uk_mobile_accepts_supported_input_forms():
    assert normalise_uk_mobile("07700 900123") == "+447700900123"
    assert normalise_uk_mobile("00447700900123") == "+447700900123"
    assert normalise_uk_mobile("+441234567890") == ""
