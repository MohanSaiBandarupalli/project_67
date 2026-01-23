from __future__ import annotations

from pathlib import Path


def test_features_use_train_only_contract():
    """
    Contract: feature builders must read TRAIN split only.
    """
    # Just a static guard: grep-style check without shell
    paths = [
        Path("src/ntg/features/build_user_features.py"),
        Path("src/ntg/features/build_item_features.py"),
        Path("src/ntg/features/build_interaction_features.py"),
    ]
    for p in paths:
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            assert "test.parquet" not in txt
            assert "val.parquet" not in txt
