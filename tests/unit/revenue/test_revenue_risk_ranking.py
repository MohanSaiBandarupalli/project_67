from __future__ import annotations

import pandas as pd


def test_revenue_risk_sorted_top_if_exists():
    from pathlib import Path

    p = Path("outputs/risk/user_risk_top.parquet")
    if not p.exists():
        return

    df = pd.read_parquet(p)
    if "revenue_at_risk_usd" in df.columns:
        vals = df["revenue_at_risk_usd"].to_list()
        assert vals == sorted(vals, reverse=True)
