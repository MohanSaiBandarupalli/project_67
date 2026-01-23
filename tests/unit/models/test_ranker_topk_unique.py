from __future__ import annotations

import pandas as pd


def test_topk_has_no_duplicates_if_exists():
    from pathlib import Path

    out = Path("outputs/recommendations/topk.parquet")
    if not out.exists():
        return

    df = pd.read_parquet(out)
    dup = df.duplicated(subset=["user_id", "item_id"]).any()
    assert not dup
