from __future__ import annotations

import pandas as pd


def test_user_features_columns_exist(write_small_splits, chdir_sandbox):
    # arrange
    import shutil
    from pathlib import Path

    # create expected structure
    (Path("data/processed/splits")).mkdir(parents=True, exist_ok=True)
    shutil.copy(write_small_splits["dir"] / "train.parquet", "data/processed/splits/train.parquet")

    # act
    import ntg.features.build_user_features as m
    m.main() if hasattr(m, "main") else m.build_user_features()

    # assert
    out = Path("data/features/user_features.parquet")
    assert out.exists()

    df = pd.read_parquet(out)
    assert "user_id" in df.columns
    assert df["user_id"].nunique() > 0
