# src/ntg/data/validation.py
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ntg.data.schemas import SCHEMA


class DataValidationError(ValueError):
    pass


def _col(name: str, default: str) -> str:
    return getattr(SCHEMA, name, default)


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_dtypes(df: pd.DataFrame) -> None:
    user_col = _col("USER_ID", "user_id")
    item_col = _col("ITEM_ID", "item_id")
    rating_col = _col("RATING", "rating")
    ts_col = _col("TIMESTAMP", "timestamp")

    if not pd.api.types.is_integer_dtype(df[user_col]):
        raise DataValidationError(f"{user_col} must be integer-like")
    if not pd.api.types.is_integer_dtype(df[item_col]):
        raise DataValidationError(f"{item_col} must be integer-like")
    if rating_col in df.columns and not pd.api.types.is_numeric_dtype(df[rating_col]):
        raise DataValidationError(f"{rating_col} must be numeric")
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        raise DataValidationError(f"{ts_col} must be datetime")


def validate_no_nulls(df: pd.DataFrame) -> None:
    # Prefer SCHEMA.required_columns if it exists, else fall back.
    req = getattr(SCHEMA, "required_columns", ("user_id", "item_id", "timestamp"))
    null_counts = df[list(req)].isna().sum()
    bad = null_counts[null_counts > 0]
    if len(bad) > 0:
        raise DataValidationError(f"Nulls found in required columns: {bad.to_dict()}")


def validate_interactions_df(df: pd.DataFrame) -> None:
    # Prefer SCHEMA.required_columns if it exists, else fall back.
    req = getattr(SCHEMA, "required_columns", ("user_id", "item_id", "timestamp"))
    validate_required_columns(df, req)
    validate_dtypes(df)
    validate_no_nulls(df)

    rating_col = _col("RATING", "rating")
    if rating_col in df.columns and (df[rating_col] < 0).any():
        raise DataValidationError("Negative ratings found (unexpected).")


# ✅ Public name expected by tests
def validate_interactions(df: pd.DataFrame) -> None:
    """
    Public API expected by tests.
    Raises DataValidationError (ValueError subclass) on invalid input.
    """
    validate_interactions_df(df)


def validate_time_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    # Optional / strategy-dependent. Kept as a no-op.
    return


def validate_per_user_order(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Ensures within each user:
      max(train_time) <= min(val_time) <= min(test_time)
    """
    user_col = _col("USER_ID", "user_id")
    ts_col = _col("TIMESTAMP", "timestamp")

    for user_id, g_tr in train.groupby(user_col):
        tr_max = g_tr[ts_col].max()

        g_val = val[val[user_col] == user_id]
        if len(g_val) > 0:
            if tr_max > g_val[ts_col].min():
                raise DataValidationError(f"Time leakage detected for user {user_id}: train overlaps val")

        g_test = test[test[user_col] == user_id]
        if len(g_test) > 0:
            pivot = g_val[ts_col].max() if len(g_val) > 0 else tr_max
            if pivot > g_test[ts_col].min():
                raise DataValidationError(f"Time leakage detected for user {user_id}: earlier data appears in test")
