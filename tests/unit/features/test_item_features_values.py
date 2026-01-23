from __future__ import annotations

import pandas as pd


def test_interaction_features_written(write_small_splits, chdir_sandbox):
    import shutil
    from pathlib import Path

    (Path("data/processed/splits")).mkdir(parents=True, exist_ok=True)
    shutil.copy(write_small_splits["dir"] / "train.parquet", "data/processed/splits/train.parquet")

    import ntg.features.build_interaction_features as m
    m.main() if hasattr(m, "main") else m.build_interaction_features()

    out = Path("data/features/interaction_features_train.parquet")
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) > 0
