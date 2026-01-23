from __future__ import annotations

import pandas as pd


def test_taste_clusters_schema(write_small_splits, chdir_sandbox):
    import shutil
    from pathlib import Path

    (Path("data/processed/splits")).mkdir(parents=True, exist_ok=True)
    shutil.copy(write_small_splits["dir"] / "train.parquet", "data/processed/splits/train.parquet")

    import ntg.graph.build_graph as bg
    bg.main()

    df = pd.read_parquet("outputs/graph/taste_clusters.parquet")
    assert {"item_id", "cluster_id"}.issubset(df.columns)
    assert df["item_id"].nunique() > 0
