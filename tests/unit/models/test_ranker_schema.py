from __future__ import annotations

import pandas as pd


def test_topk_schema_if_exists():
    """
    Unit-level contract: if topk output exists, schema must be correct.
    """
    from pathlib import Path

    out = Path("outputs/recommendations/topk.parquet")
    if not out.exists():
        return

    df = pd.read_parquet(out)
    assert {"user_id", "item_id"}.issubset(df.columns)
