"""Public compatibility module for the ATC Roster application.

Application assembly now lives in :mod:`atcroster.application`; this module
remains the stable import target for WSGI, CLI scripts, tests, and deployments.
"""

import sys

from atcroster import application as _application

# Preserve a single module object so existing callers that monkeypatch
# ``app`` continue to affect the running application implementation.
sys.modules[__name__] = _application
