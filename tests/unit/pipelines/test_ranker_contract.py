from __future__ import annotations

from pathlib import Path


def test_day4_ranker_outputs_exist_if_present():
    base = Path("outputs/recommendations")
    if not base.exists():
        return
    assert (base / "topk.parquet").exists()
