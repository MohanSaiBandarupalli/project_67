from __future__ import annotations

import pytest


def test_time_module_imports():
    import ntg.common.time as t
    assert t is not None


def test_now_fn_if_exists():
    import ntg.common.time as t
    fn = getattr(t, "utcnow", None) or getattr(t, "now_utc", None)
    if fn is None:
        pytest.skip("no utc time helper found (ok)")
    out = fn()
    assert out is not None
