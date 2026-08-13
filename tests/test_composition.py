from __future__ import annotations

import pytest

from atcroster.composition import DeferredReference


def test_deferred_reference_requires_configuration_and_is_single_assignment():
    reference: DeferredReference[str] = DeferredReference("example")

    with pytest.raises(RuntimeError, match="not configured: example"):
        reference.get()

    reference.set("configured")
    assert reference.get() == "configured"

    with pytest.raises(RuntimeError, match="already configured: example"):
        reference.set("replacement")
