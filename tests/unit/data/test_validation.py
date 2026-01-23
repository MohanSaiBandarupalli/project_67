from __future__ import annotations

import pandas as pd
import pytest


def test_validation_rejects_missing_columns(small_interactions_df):
    from ntg.data.validation import validate_interactions

    bad = small_interactions_df.drop(columns=["rating"])
    with pytest.raises(Exception):
        validate_interactions(bad)


def test_validation_accepts_valid_df(small_interactions_df):
    from ntg.data.validation import validate_interactions

    validate_interactions(small_interactions_df)  # should not raise


def test_validation_rejects_non_datetime_timestamp(small_interactions_df):
    from ntg.data.validation import validate_interactions

    bad = small_interactions_df.copy()
    bad["timestamp"] = bad["timestamp"].astype(str)
    with pytest.raises(Exception):
        validate_interactions(bad)
