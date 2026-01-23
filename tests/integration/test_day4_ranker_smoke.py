from __future__ import annotations

from pathlib import Path
import pytest


@pytest.mark.integration
def test_day4_ranker_smoke(has_movielens_data):
    if not has_movielens_data:
        pytest.skip("MovieLens data not present")

    import ntg.models.ranker as r
    if not hasattr(r, "main"):
        pytest.skip("ranker.main not available")
    r.main()

    assert Path("outputs/recommendations/topk.parquet").exists()
    assert Path("reports/ranking_metrics.json").exists()
