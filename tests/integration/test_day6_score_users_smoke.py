from __future__ import annotations

from pathlib import Path
import pytest


@pytest.mark.integration
def test_day6_score_users_smoke(has_movielens_data):
    if not has_movielens_data:
        pytest.skip("MovieLens data not present")

    import ntg.pipelines.score_users as p
    p.main()

    assert Path("outputs/risk/churn_scores.parquet").exists()
    assert Path("reports/revenue_risk_summary.json").exists()
