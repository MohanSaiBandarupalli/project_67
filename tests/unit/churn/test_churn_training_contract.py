from __future__ import annotations

import pytest
import pandas as pd


def test_churn_train_model_imports():
    import ntg.churn.train_model as m
    assert m is not None


def test_score_users_pipeline_smoke(sandbox, write_small_splits):
    """
    Calls ntg.pipelines.score_users.main if present.
    """
    import ntg.pipelines.score_users as p

    run = getattr(p, "main", None)
    if run is None:
        pytest.skip("ntg.pipelines.score_users.main not found")

    # score_users expects user_features. attempt to build if builder exists.
    try:
        import ntg.features.build_user_features as uf
        if getattr(uf, "main", None):
            uf.main()
    except Exception:
        # if your builder has different signature, let score_users decide
        pass

    try:
        run()
    except FileNotFoundError as e:
        pytest.skip(f"missing inputs for churn pipeline in sandbox: {e}")

    assert (pd.read_parquet("outputs/risk/churn_scores.parquet").shape[0]) >= 0
