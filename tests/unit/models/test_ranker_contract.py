from __future__ import annotations

import pytest
import pandas as pd


def test_ranker_module_imports():
    import ntg.models.ranker as m
    assert m is not None


def test_ranker_produces_topk(sandbox, write_small_splits):
    """
    Runs ranker if main exists.
    Requires:
      - splits exist (train/val/test)
      - features exist (some pipelines generate them, we may skip if missing)
    """
    import ntg.models.ranker as m

    run = getattr(m, "main", None)
    if run is None:
        pytest.skip("ntg.models.ranker.main not found")

    # If your ranker expects feature files, you may need to run feature builders first.
    # We'll try; if it crashes due to missing inputs, we skip (contract-based).
    try:
        run()
    except FileNotFoundError as e:
        pytest.skip(f"ranker inputs missing in this sandbox: {e}")

    out = "outputs/recommendations/topk.parquet"
    df = pd.read_parquet(out)
    assert {"user_id", "item_id"}.issubset(df.columns)
    assert len(df) > 0
