from __future__ import annotations

import pytest


def test_metrics_module_imports():
    import ntg.evaluation.metrics as m
    assert m is not None


def test_ndcg_map_helpers_if_exist():
    import ntg.evaluation.metrics as m

    ndcg = getattr(m, "ndcg_at_k", None)
    mapk = getattr(m, "map_at_k", None)
    if ndcg is None or mapk is None:
        pytest.skip("ndcg_at_k/map_at_k not found (ok)")

    # perfect ranking
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    assert ndcg(y_true, y_score, k=4) == pytest.approx(1.0, abs=1e-9)
    assert mapk(y_true, y_score, k=4) == pytest.approx(1.0, abs=1e-9)

    # worst ranking
    y_score_bad = [0.1, 0.2, 0.8, 0.9]
    assert ndcg(y_true, y_score_bad, k=4) <= 0.5
