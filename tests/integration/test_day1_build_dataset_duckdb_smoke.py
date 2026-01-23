from __future__ import annotations

import pytest


@pytest.mark.integration
def test_day1_pipeline_imports():
    import ntg.pipelines.build_dataset_duckdb as m
    assert m is not None


@pytest.mark.integration
def test_day1_pipeline_smoke_runs_if_data_present(project_root):
    """
    Only runs if MovieLens exists locally (not in CI).
    """
    data_dir = project_root / "data/external/movielens"
    if not data_dir.exists():
        pytest.skip("no external data in CI")

    import ntg.pipelines.build_dataset_duckdb as m
    run = getattr(m, "main", None)
    if run is None:
        pytest.skip("build_dataset_duckdb.main not found")

    run()
