from __future__ import annotations

import pytest
import pandas as pd


def test_item_features_builder_imports():
    import ntg.features.build_item_features as m
    assert m is not None


def test_item_features_output_schema_smoke(sandbox, write_small_splits):
    import ntg.features.build_item_features as m

    run = getattr(m, "main", None) or getattr(m, "build_item_features", None)
    if run is None:
        pytest.skip("item feature builder doesn't expose main/build_item_features")

    run()
    df = pd.read_parquet("data/features/item_features.parquet")
    assert "item_id" in df.columns
    assert df["item_id"].notna().all()
    assert df["item_id"].nunique() > 0
