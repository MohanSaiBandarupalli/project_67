from __future__ import annotations

import pytest
import pandas as pd


def _import_splits():
    import ntg.data.splits as splits
    return splits


def test_per_user_split_exists():
    splits = _import_splits()
    assert hasattr(splits, "time_split_per_user")


def test_time_split_config_backcompat_exists():
    splits = _import_splits()
    # CI earlier failed because TimeSplitConfig was missing.
    assert hasattr(splits, "TimeSplitConfig") or hasattr(splits, "PerUserTimeSplitConfig")


@pytest.mark.parametrize("train_frac,val_frac", [(0.7, 0.15), (0.6, 0.2), (0.8, 0.1)])
def test_per_user_split_fractions_valid(small_interactions_df: pd.DataFrame, train_frac: float, val_frac: float):
    splits = _import_splits()
    Cfg = getattr(splits, "TimeSplitConfig", getattr(splits, "PerUserTimeSplitConfig"))
    cfg = Cfg(train_frac=train_frac, val_frac=val_frac, min_interactions=5)

    train, val, test, meta = splits.time_split_per_user(small_interactions_df, cfg)

    assert len(train) + len(val) + len(test) == len(small_interactions_df)
    assert meta["train_frac"] == pytest.approx(train_frac)
    assert meta["val_frac"] == pytest.approx(val_frac)


def test_per_user_split_chronological_per_user(small_interactions_df: pd.DataFrame):
    splits = _import_splits()
    Cfg = getattr(splits, "TimeSplitConfig", getattr(splits, "PerUserTimeSplitConfig"))
    cfg = Cfg(train_frac=0.7, val_frac=0.15, min_interactions=5)

    train, val, test, _ = splits.time_split_per_user(small_interactions_df, cfg)

    # For each user: max(train.ts) <= min(val.ts) <= min(test.ts) (when those exist)
    for uid in small_interactions_df["user_id"].unique():
        tr = train[train["user_id"] == uid]
        va = val[val["user_id"] == uid]
        te = test[test["user_id"] == uid]

        if len(va) > 0:
            assert tr["timestamp"].max() <= va["timestamp"].min()
        if len(te) > 0 and len(va) > 0:
            assert va["timestamp"].max() <= te["timestamp"].min()
        if len(te) > 0 and len(va) == 0:
            # edge case: val can be empty, but test should still be >= train
            assert tr["timestamp"].max() <= te["timestamp"].min()


def test_per_user_min_interactions_behavior(small_interactions_df: pd.DataFrame):
    splits = _import_splits()
    Cfg = getattr(splits, "TimeSplitConfig", getattr(splits, "PerUserTimeSplitConfig"))
    cfg = Cfg(train_frac=0.7, val_frac=0.15, min_interactions=5)

    train, val, test, _ = splits.time_split_per_user(small_interactions_df, cfg)

    # user_id=2 has only 4 interactions -> should all go to train, no val/test
    assert len(val[val["user_id"] == 2]) == 0
    assert len(test[test["user_id"] == 2]) == 0
    assert len(train[train["user_id"] == 2]) == 4


def test_per_user_split_is_deterministic(small_interactions_df: pd.DataFrame):
    splits = _import_splits()
    Cfg = getattr(splits, "TimeSplitConfig", getattr(splits, "PerUserTimeSplitConfig"))
    cfg = Cfg(train_frac=0.7, val_frac=0.15, min_interactions=5)

    a = splits.time_split_per_user(small_interactions_df, cfg)
    b = splits.time_split_per_user(small_interactions_df, cfg)

    for i in range(3):
        assert a[i].reset_index(drop=True).equals(b[i].reset_index(drop=True))


@pytest.mark.parametrize(
    "train_frac,val_frac",
    [(1.0, 0.0), (0.0, 0.2), (0.8, 0.25), (-0.1, 0.1)],
)
def test_per_user_split_invalid_config_raises(small_interactions_df: pd.DataFrame, train_frac: float, val_frac: float):
    splits = _import_splits()
    Cfg = getattr(splits, "TimeSplitConfig", getattr(splits, "PerUserTimeSplitConfig"))
    cfg = Cfg(train_frac=train_frac, val_frac=val_frac, min_interactions=5)
    with pytest.raises(ValueError):
        splits.time_split_per_user(small_interactions_df, cfg)
