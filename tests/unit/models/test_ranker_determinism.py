from __future__ import annotations

import pandas as pd


def test_ranker_deterministic_if_runnable(chdir_sandbox, write_small_splits):
    """
    If module provides a callable main, running twice should yield same output.
    """
    import shutil
    from pathlib import Path

    (Path("data/processed/splits")).mkdir(parents=True, exist_ok=True)
    shutil.copy(write_small_splits["dir"] / "train.parquet", "data/processed/splits/train.parquet")
    shutil.copy(write_small_splits["dir"] / "val.parquet", "data/processed/splits/val.parquet")
    shutil.copy(write_small_splits["dir"] / "test.parquet", "data/processed/splits/test.parquet")

    import ntg.models.ranker as r
    if not hasattr(r, "main"):
        return

    r.main()
    a = pd.read_parquet("outputs/recommendations/topk.parquet").sort_values(["user_id", "item_id"]).reset_index(drop=True)

    r.main()
    b = pd.read_parquet("outputs/recommendations/topk.parquet").sort_values(["user_id", "item_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(a, b, check_dtype=False)
