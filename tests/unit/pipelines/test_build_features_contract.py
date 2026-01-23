from __future__ import annotations

from pathlib import Path


def test_day2_feature_outputs_exist_if_present():
    base = Path("data/features")
    if not base.exists():
        return
    # if folder exists, ensure at least one parquet present
    assert any(p.suffix == ".parquet" for p in base.glob("*.parquet"))
