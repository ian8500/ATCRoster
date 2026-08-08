# Sinch MessageMedia SMS handover

## Safe cutover

1. In Sinch MessageMedia, create an API key with access to the Messages API.
2. Add `MESSAGEMEDIA_API_KEY`, `MESSAGEMEDIA_API_SECRET` and a verified
   `MESSAGEMEDIA_FALLBACK_SENDER` to the production environment.
3. Set `SMS_PROVIDER=messagemedia`, deploy, then send a controlled test to a
   consenting staff member and confirm the audit entry shows `submitted`.
4. A Watch Manager registers their UK handset in their profile, then verifies
   it in Sinch Engage under **Settings → Numbers → Active → My own numbers**.
   A Unit Administrator records the successful verification in SMS audit.
5. Confirm replies reach the manager's handset; they intentionally do not enter
   ATCRoster. Re-verification is due before the recorded expiry (normally 12
   months).
6. In Sinch MessageMedia **Settings → API → Webhooks**, create a delivery
   report webhook to `https://www.atcroster.com/webhooks/messagemedia/delivery`
   and configure the `X-ATCRoster-Webhook-Token` header with the matching
   environment value. Delivery state then updates the SMS audit.

## Privacy and operations

ATCRoster retains SMS audit records for operational accountability. The audit
contains sender, recipient, message content, provider identifier and delivery
state; it never stores verification codes or API secrets. Use operational SMS
only where there is a lawful basis and recipient consent/process coverage.

Failed provider submissions are reported to the sender and are not presented as
delivered. The legacy provider can only be re-enabled deliberately with
`SMS_PROVIDER=twilio` while its credentials remain configured.
