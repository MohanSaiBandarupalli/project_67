from __future__ import annotations

from pathlib import Path


def test_ranker_does_not_read_test_split_code_guard():
    p = Path("src/ntg/models/ranker.py")
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8")
    assert "test.parquet" not in txt
