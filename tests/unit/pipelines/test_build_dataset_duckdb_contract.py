from __future__ import annotations

from pathlib import Path


def test_day1_contract_outputs_exist_if_run_locally():
    """
    Contract-only: if user ran Day-1, these files must exist.
    """
    splits = Path("data/processed/splits")
    if not splits.exists():
        return
    assert (splits / "train.parquet").exists()
    assert (splits / "val.parquet").exists()
    assert (splits / "test.parquet").exists()
