from __future__ import annotations

from pathlib import Path


def test_day6_outputs_exist_if_present():
    base = Path("outputs/risk")
    if not base.exists():
        return
    assert (base / "churn_scores.parquet").exists()
