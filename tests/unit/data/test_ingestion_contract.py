# tests/unit/data/test_splits_contract.py
from __future__ import annotations

import pandas as pd
import pytest

from ntg.data.schemas import SCHEMA
from ntg.data import splits as splits_mod


def toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SCHEMA.USER_ID: [1, 1, 1, 2, 2, 2, 2, 3, 3],
            SCHEMA.ITEM_ID: [10, 11, 12, 10, 13, 14, 15, 20, 21],
            SCHEMA.TIMESTAMP: [1, 2, 3, 1, 2, 3, 4, 10, 11],
            "rating": [4, 5, 3, 4, 2, 5, 4, 5, 4],
        }
    )


def test_backward_compat_time_split_config_exists():
    assert hasattr(splits_mod, "TimeSplitConfig"), "Expected TimeSplitConfig for backward compatibility"


def test_backward_compat_time_split_per_user_exists():
    assert hasattr(splits_mod, "time_split_per_user"), "Expected time_split_per_user for backward compatibility"


def test_per_user_split_deterministic():
    df = toy_df()
    cfg = splits_mod.TimeSplitConfig(train_frac=0.6, val_frac=0.2, min_interactions=3)
    a = splits_mod.time_split_per_user(df, cfg)
    b = splits_mod.time_split_per_user(df, cfg)
    for i in range(3):
        pd.testing.assert_frame_equal(a[i], b[i])


def test_per_user_ordering_no_leakage():
    df = toy_df()
    cfg = splits_mod.TimeSplitConfig(train_frac=0.5, val_frac=0.25, min_interactions=3)
    tr, va, te, _ = splits_mod.time_split_per_user(df, cfg)

    for uid in df[SCHEMA.USER_ID].unique():
        tr_ts = tr.loc[tr[SCHEMA.USER_ID] == uid, SCHEMA.TIMESTAMP]
        va_ts = va.loc[va[SCHEMA.USER_ID] == uid, SCHEMA.TIMESTAMP]
        te_ts = te.loc[te[SCHEMA.USER_ID] == uid, SCHEMA.TIMESTAMP]
        if len(va_ts) > 0:
            assert tr_ts.max() <= va_ts.min()
        if len(te_ts) > 0 and len(va_ts) > 0:
            assert va_ts.max() <= te_ts.min()


def test_min_interactions_all_train():
    df = toy_df()
    cfg = splits_mod.TimeSplitConfig(train_frac=0.6, val_frac=0.2, min_interactions=100)
    tr, va, te, _ = splits_mod.time_split_per_user(df, cfg)
    assert len(tr) == len(df)
    assert len(va) == 0
    assert len(te) == 0


def test_invalid_train_frac_raises():
    df = toy_df()
    with pytest.raises(ValueError):
        splits_mod.time_split_per_user(df, splits_mod.TimeSplitConfig(train_frac=1.0, val_frac=0.1))


def test_invalid_val_frac_raises():
    df = toy_df()
    with pytest.raises(ValueError):
        splits_mod.time_split_per_user(df, splits_mod.TimeSplitConfig(train_frac=0.8, val_frac=0.3))


def test_empty_df_returns_empty_parts():
    df = toy_df().iloc[0:0].copy()
    cfg = splits_mod.TimeSplitConfig()
    tr, va, te, meta = splits_mod.time_split_per_user(df, cfg)
    assert len(tr) == 0 and len(va) == 0 and len(te) == 0
    assert meta["n_total"] == 0.0


def test_split_columns_preserved():
    df = toy_df()
    cfg = splits_mod.TimeSplitConfig()
    tr, va, te, _ = splits_mod.time_split_per_user(df, cfg)
    assert set(tr.columns) == set(df.columns)
    assert set(va.columns) == set(df.columns)
    assert set(te.columns) == set(df.columns)


def test_global_split_exists_if_enabled():
    assert hasattr(splits_mod, "time_split_global"), "Expected time_split_global in splits module"
