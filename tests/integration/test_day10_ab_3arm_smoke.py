from __future__ import annotations

from pathlib import Path
import pytest


@pytest.mark.integration
def test_day10_ab_3arm_smoke(has_movielens_data):
    if not has_movielens_data:
        pytest.skip("MovieLens data not present")

    import ntg.pipelines.run_experiment_3arm as e
    e.main()

    assert Path("reports/ab_results_3arm.json").exists()
    assert Path("reports/ab_experiment_design_3arm.md").exists()
