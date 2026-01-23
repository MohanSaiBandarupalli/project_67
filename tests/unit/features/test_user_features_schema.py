from __future__ import annotations

import pytest
import pandas as pd


def test_user_features_builder_imports():
    import ntg.features.build_user_features as m
    assert m is not None


def test_user_features_output_schema_smoke(sandbox, write_small_splits):
    """
    Runs the user-features builder if it exposes a callable.
    Otherwise, skip.
    """
    import ntg.features.build_user_features as m

    run = getattr(m, "main", None) or getattr(m, "build_user_features", None)
    if run is None:
        pytest.skip("user feature builder doesn't expose main/build_user_features")

    run()  # should write data/features/user_features.parquet

    df = pd.read_parquet("data/features/user_features.parquet")
    assert "user_id" in df.columns
    assert df["user_id"].notna().all()
    assert df["user_id"].nunique() > 0
