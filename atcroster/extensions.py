"""Extension ownership for incremental migration out of the legacy module.

Database and login extensions remain in ``app.py`` until their model and
session-routing dependencies are extracted. New extensions belong here.
"""
