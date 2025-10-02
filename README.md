# ATC Roster

Flask + SQLAlchemy app for rostering ATCOs.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
FLASK_APP=app.py FLASK_ENV=development flask run

## SMS configuration

Overtime SMS notifications use Twilio's REST API. Set the following environment variables before starting the app:

* `TWILIO_ACCOUNT_SID`
* `TWILIO_AUTH_TOKEN`
* `TWILIO_FROM_NUMBER`

If these are not configured the send button on the overtime page will be disabled.
