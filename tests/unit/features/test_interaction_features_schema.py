from __future__ import annotations

import pytest
import pandas as pd


def test_interaction_features_builder_imports():
    import ntg.features.build_interaction_features as m
    assert m is not None


def test_interaction_features_output_exists(sandbox, write_small_splits):
    import ntg.features.build_interaction_features as m

    run = getattr(m, "main", None) or getattr(m, "build_interaction_features", None)
    if run is None:
        pytest.skip("interaction feature builder doesn't expose main/build_interaction_features")

    run()
    p = "data/features/interaction_features_train.parquet"
    df = pd.read_parquet(p)
    # minimal contract: should have user_id + item_id at least
    assert "user_id" in df.columns
    assert "item_id" in df.columns
    assert len(df) > 0
