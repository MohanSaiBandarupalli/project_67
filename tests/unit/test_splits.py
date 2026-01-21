# tests/unit/test_splits.py
import pandas as pd

from ntg.data.splits import TimeSplitConfig, time_split_per_user
from ntg.data.schemas import SCHEMA
from ntg.data.validation import validate_per_user_order


def _toy_df() -> pd.DataFrame:
    # user 1: 10 interactions
    # user 2: 4 interactions (should all go to train when min_interactions=5)
    df1 = pd.DataFrame({
        SCHEMA.USER_ID: [1] * 10,
        SCHEMA.ITEM_ID: list(range(10)),
        SCHEMA.RATING: [4.0] * 10,
        SCHEMA.TIMESTAMP: pd.date_range("2020-01-01", periods=10, freq="D"),
    })
    df2 = pd.DataFrame({
        SCHEMA.USER_ID: [2] * 4,
        SCHEMA.ITEM_ID: list(range(100, 104)),
        SCHEMA.RATING: [3.0] * 4,
        SCHEMA.TIMESTAMP: pd.date_range("2021-01-01", periods=4, freq="D"),
    })
    return pd.concat([df1, df2], ignore_index=True)


def test_time_split_per_user_order_and_min_interactions():
    df = _toy_df()
    cfg = TimeSplitConfig(train_frac=0.7, val_frac=0.2, min_interactions=5)

    train, val, test, meta = time_split_per_user(df, cfg)

    # user 2 should be train-only (min_interactions behavior)
    assert (val[SCHEMA.USER_ID] == 2).sum() == 0
    assert (test[SCHEMA.USER_ID] == 2).sum() == 0
    assert (train[SCHEMA.USER_ID] == 2).sum() == 4

    # leakage check (per-user)
    validate_per_user_order(train, val, test)

    # metadata sanity
    assert meta["n_total"] == float(len(df))
    assert meta["n_train"] + meta["n_val"] + meta["n_test"] == float(len(df))


def test_split_is_deterministic():
    df = _toy_df()
    cfg = TimeSplitConfig(train_frac=0.7, val_frac=0.2, min_interactions=5)

    t1, v1, te1, _ = time_split_per_user(df, cfg)
    t2, v2, te2, _ = time_split_per_user(df, cfg)

    # Compare content (order and rows)
    pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
    pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))
    pd.testing.assert_frame_equal(te1.reset_index(drop=True), te2.reset_index(drop=True))
