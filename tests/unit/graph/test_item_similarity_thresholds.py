from __future__ import annotations

import pandas as pd


def test_item_similarity_respects_min_cosine(write_small_splits, chdir_sandbox):
    import shutil
    from pathlib import Path

    (Path("data/processed/splits")).mkdir(parents=True, exist_ok=True)
    shutil.copy(write_small_splits["dir"] / "train.parquet", "data/processed/splits/train.parquet")

    import ntg.graph.item_similarity as m
    cfg = m.ItemSimConfig(min_cosine=0.10, min_item_support=1, min_cooc=1, topk_per_item=50)
    m.build_item_similarity(cfg)

    df = pd.read_parquet("outputs/graph/item_item.parquet")
    assert (df["cosine"] >= 0.10 - 1e-12).all()
