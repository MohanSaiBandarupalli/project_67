from __future__ import annotations

import pandas as pd


def test_churn_prob_in_0_1_if_exists():
    from pathlib import Path

    p = Path("outputs/risk/churn_scores.parquet")
    if not p.exists():
        return
    df = pd.read_parquet(p)
    assert ((df["churn_prob"] >= 0.0) & (df["churn_prob"] <= 1.0)).all()
