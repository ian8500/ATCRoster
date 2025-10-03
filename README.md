# ATC Roster

Flask + SQLAlchemy app for rostering ATCOs.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
FLASK_APP=app.py FLASK_ENV=development flask run
```

## Styling and colour themes

The application ships with a bespoke dark palette defined in
[`static/styles.css`](static/styles.css). The Bootstrap bundle is loaded in
dark mode via `data-bs-theme="dark"` on the root `<html>` element and the
`<meta name="color-scheme" content="dark">` hint in
[`templates/base.html`](templates/base.html). These settings do not block you
from applying a new colour scheme, but they do tell the browser and Bootstrap
to prefer dark-friendly defaults for controls.

To roll out a different palette you can either override the CSS custom
properties in `styles.css` or introduce an alternative `[data-theme]`
modifier—just remember to update the `data-bs-theme` attribute if you want
Bootstrap's components to follow suit (e.g. switch it to `light` for a light
look). Removing or changing the `color-scheme` meta tag is also safe if you
need to support light form controls.

## SMS configuration

Overtime SMS notifications use Twilio's REST API. Set the following environment variables before starting the app:

* `TWILIO_ACCOUNT_SID`
* `TWILIO_AUTH_TOKEN`
* `TWILIO_FROM_NUMBER`

If these are not configured the send button on the overtime page will be disabled.
