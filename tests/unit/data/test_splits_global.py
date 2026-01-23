from __future__ import annotations

import pytest
import pandas as pd


def test_global_split_exists():
    import ntg.data.splits as splits
    assert hasattr(splits, "time_split_global")
    assert hasattr(splits, "GlobalTimeSplitConfig")


@pytest.mark.parametrize("train_frac,val_frac", [(0.7, 0.15), (0.6, 0.2), (0.8, 0.1)])
def test_global_split_sizes(small_interactions_df: pd.DataFrame, train_frac: float, val_frac: float):
    import ntg.data.splits as splits

    cfg = splits.GlobalTimeSplitConfig(train_frac=train_frac, val_frac=val_frac)
    train, val, test, meta = splits.time_split_global(small_interactions_df, cfg)

    assert len(train) + len(val) + len(test) == len(small_interactions_df)
    assert meta["train_frac"] == pytest.approx(train_frac)
    assert meta["val_frac"] == pytest.approx(val_frac)


def test_global_split_is_chronological(small_interactions_df: pd.DataFrame):
    import ntg.data.splits as splits

    cfg = splits.GlobalTimeSplitConfig(train_frac=0.7, val_frac=0.15)
    train, val, test, _ = splits.time_split_global(small_interactions_df, cfg)

    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


@pytest.mark.parametrize(
    "train_frac,val_frac",
    [(1.0, 0.0), (0.0, 0.2), (0.8, 0.25), (-0.1, 0.1)],
)
def test_global_split_invalid_raises(small_interactions_df: pd.DataFrame, train_frac: float, val_frac: float):
    import ntg.data.splits as splits

    cfg = splits.GlobalTimeSplitConfig(train_frac=train_frac, val_frac=val_frac)
    with pytest.raises(ValueError):
        splits.time_split_global(small_interactions_df, cfg)
