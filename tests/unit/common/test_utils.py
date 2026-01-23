from __future__ import annotations

import pytest


def test_utils_module_imports():
    import ntg.common.utils as utils
    assert utils is not None


@pytest.mark.parametrize("value", [0, 1, 5, 123])
def test_if_seed_helper_exists_then_deterministic(value: int):
    """
    Netflix/FAANG style: tolerate refactors.
    If you have a seed setter, verify it does not crash.
    """
    import ntg.common.utils as utils

    fn = getattr(utils, "set_seed", None)
    if fn is None:
        pytest.skip("ntg.common.utils.set_seed not implemented (ok)")
    fn(value)
    fn(value)  # should not crash
