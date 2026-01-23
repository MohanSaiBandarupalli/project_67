from __future__ import annotations

import pandas as pd


def test_churn_scores_schema_if_exists():
    from pathlib import Path

    p = Path("outputs/risk/churn_scores.parquet")
    if not p.exists():
        return
    df = pd.read_parquet(p)
    assert {"user_id", "churn_prob"}.issubset(df.columns)
