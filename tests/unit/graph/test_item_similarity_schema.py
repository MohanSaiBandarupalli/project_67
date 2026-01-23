from __future__ import annotations

import pandas as pd


def test_item_item_schema(write_small_splits, chdir_sandbox):
    import shutil
    from pathlib import Path

    (Path("data/processed/splits")).mkdir(parents=True, exist_ok=True)
    shutil.copy(write_small_splits["dir"] / "train.parquet", "data/processed/splits/train.parquet")

    import ntg.graph.item_similarity as m
    m.build_item_similarity(m.ItemSimConfig())

    out = Path("outputs/graph/item_item.parquet")
    assert out.exists()

    df = pd.read_parquet(out)
    expected = {"src_item", "dst_item", "cosine", "cooc_cnt", "src_cnt", "dst_cnt"}
    assert expected.issubset(set(df.columns))
