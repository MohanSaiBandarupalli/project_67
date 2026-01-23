from __future__ import annotations

import numpy as np
import pytest


def test_metrics_handle_empty_gracefully():
    import ntg.evaluation.metrics as m

    if hasattr(m, "precision_at_k"):
        assert m.precision_at_k([], [], k=10) == 0.0


def test_metrics_handle_perfect_case():
    import ntg.evaluation.metrics as m

    if hasattr(m, "precision_at_k"):
        assert m.precision_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0
