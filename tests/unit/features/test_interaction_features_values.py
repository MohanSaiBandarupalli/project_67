from __future__ import annotations

import pandas as pd


def test_item_features_columns_exist(write_small_splits, chdir_sandbox):
    import shutil
    from pathlib import Path

    (Path("data/processed/splits")).mkdir(parents=True, exist_ok=True)
    shutil.copy(write_small_splits["dir"] / "train.parquet", "data/processed/splits/train.parquet")

    import ntg.features.build_item_features as m
    m.main() if hasattr(m, "main") else m.build_item_features()

    out = Path("data/features/item_features.parquet")
    assert out.exists()

    df = pd.read_parquet(out)
    assert "item_id" in df.columns
    assert df["item_id"].nunique() > 0
