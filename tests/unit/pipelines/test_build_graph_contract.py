from __future__ import annotations

from pathlib import Path


def test_day3_graph_outputs_exist_if_present():
    base = Path("outputs/graph")
    if not base.exists():
        return
    assert (base / "item_item.parquet").exists()
