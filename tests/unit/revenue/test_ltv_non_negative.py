from __future__ import annotations

import pandas as pd


def test_ltv_non_negative_if_exists():
    from pathlib import Path

    p = Path("outputs/risk/user_ltv.parquet")
    if not p.exists():
        return
    df = pd.read_parquet(p)
    if "ltv_usd" in df.columns:
        assert (df["ltv_usd"] >= 0).all()
