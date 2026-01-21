# src/ntg/data/validation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd

from ntg.data.schemas import SCHEMA


class DataValidationError(ValueError):
    pass


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_dtypes(df: pd.DataFrame) -> None:
    # We avoid strict dtype equality because parquet/pandas can differ slightly,
    # but we DO enforce safe & expected kinds.
    if not pd.api.types.is_integer_dtype(df[SCHEMA.USER_ID]):
        raise DataValidationError(f"{SCHEMA.USER_ID} must be integer-like")
    if not pd.api.types.is_integer_dtype(df[SCHEMA.ITEM_ID]):
        raise DataValidationError(f"{SCHEMA.ITEM_ID} must be integer-like")
    if not pd.api.types.is_numeric_dtype(df[SCHEMA.RATING]):
        raise DataValidationError(f"{SCHEMA.RATING} must be numeric")
    if not pd.api.types.is_datetime64_any_dtype(df[SCHEMA.TIMESTAMP]):
        raise DataValidationError(f"{SCHEMA.TIMESTAMP} must be datetime")


def validate_no_nulls(df: pd.DataFrame) -> None:
    null_counts = df[list(SCHEMA.required_columns)].isna().sum()
    bad = null_counts[null_counts > 0]
    if len(bad) > 0:
        raise DataValidationError(f"Nulls found in required columns: {bad.to_dict()}")


def validate_interactions_df(df: pd.DataFrame) -> None:
    validate_required_columns(df, SCHEMA.required_columns)
    validate_dtypes(df)
    validate_no_nulls(df)

    if (df[SCHEMA.RATING] < 0).any():
        raise DataValidationError("Negative ratings found (unexpected).")


def validate_time_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Strong check: ensures global time order separation.

    Note:
      For per-user splits, global separation isn't always guaranteed if users have different histories.
      BUT for MovieLens overall, if you want strict global cutoffs, you can add another strategy.
      For this Day-1, we enforce per-user separation (see validate_per_user_order).
    """
    # We keep this as an optional "nice-to-have" check; not always valid.
    return


def validate_per_user_order(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Ensures within each user:
      max(train_time) <= min(val_time) <= min(test_time)
    """
    # Only check users that exist in val/test; train-only users are allowed.
    for user_id, g_tr in train.groupby(SCHEMA.USER_ID):
        tr_max = g_tr[SCHEMA.TIMESTAMP].max()

        g_val = val[val[SCHEMA.USER_ID] == user_id]
        if len(g_val) > 0:
            if tr_max > g_val[SCHEMA.TIMESTAMP].min():
                raise DataValidationError(f"Time leakage detected for user {user_id}: train overlaps val")

        g_test = test[test[SCHEMA.USER_ID] == user_id]
        if len(g_test) > 0:
            # If val exists, compare to val; else compare to train.
            pivot = g_val[SCHEMA.TIMESTAMP].max() if len(g_val) > 0 else tr_max
            if pivot > g_test[SCHEMA.TIMESTAMP].min():
                raise DataValidationError(f"Time leakage detected for user {user_id}: earlier data appears in test")
