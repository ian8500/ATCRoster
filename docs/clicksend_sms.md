# ClickSend SMS

ATCO Roster sends SMS as `ATCO Roster → ClickSend REST API → recipient`. The
application uses `POST https://rest.clicksend.com/v3/sms/send` with HTTP Basic
authentication; the ClickSend username and API key are platform secrets.

## Configuration

Set these in local secure environment configuration or Railway Variables:

```text
CLICK_SEND_USERNAME=
CLICK_SEND_API_KEY=
CLICK_SEND_DEFAULT_SENDER=
```

`CLICK_SEND_DEFAULT_SENDER` must be a UK mobile verified in ClickSend as an
Own Number. Do not put any of these values in Git. Each airport may configure
its own verified sender in SMS settings; that unit-scoped configuration takes
precedence over the platform default, so another airport's sender cannot be
selected.

Verify numbers in ClickSend before entering them in ATCO Roster. The app does
not perform verification and never substitutes a different sender when one is
rejected. Own Number replies go to the underlying handset; ATCO Roster has no
inbound SMS processing.

## Testing and operations

An authorised Unit Administrator can use **Admin → SMS audit → Send test SMS**.
It is CSRF-protected, unit-scoped, recipient-validated and rate-limited. The
test says: `ATCO Roster SMS test. ClickSend messaging is configured correctly.`

Normal tests mock HTTP and never send SMS. A live test is intentionally manual
and requires a deliberately supplied test number. For Railway, add the three
Variables above, deploy, then run one controlled test before retiring legacy
provider variables.

## Troubleshooting and rotation

Check credentials for 401/403 responses, Own Number verification for sender
rejections, recipient E.164 format, account balance/country enablement, and
ClickSend throttling/outages. Timeout and 5xx responses are not retried because
the provider might have accepted the SMS. Rotate `CLICK_SEND_API_KEY` by adding
the new Railway/local secret, deploying, testing, then revoking the old key in
ClickSend. Credentials are never shown in HTML, JavaScript, logs or SMS audit.
