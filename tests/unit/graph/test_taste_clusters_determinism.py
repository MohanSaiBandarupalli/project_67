from __future__ import annotations

import pandas as pd


def test_taste_clusters_deterministic(write_small_splits, chdir_sandbox):
    import shutil
    from pathlib import Path

    (Path("data/processed/splits")).mkdir(parents=True, exist_ok=True)
    shutil.copy(write_small_splits["dir"] / "train.parquet", "data/processed/splits/train.parquet")

    import ntg.graph.build_graph as bg

    bg.main()
    a = pd.read_parquet("outputs/graph/taste_clusters.parquet").sort_values(["item_id"]).reset_index(drop=True)

    # run again
    bg.main()
    b = pd.read_parquet("outputs/graph/taste_clusters.parquet").sort_values(["item_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(a, b, check_dtype=False)
