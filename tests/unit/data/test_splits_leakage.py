from __future__ import annotations

import pandas as pd


def test_no_overlap_between_splits(write_small_splits):
    tr = write_small_splits["train"]
    va = write_small_splits["val"]
    te = write_small_splits["test"]

    # define unique interaction key
    def keys(df: pd.DataFrame):
        return set(zip(df["user_id"], df["item_id"], df["timestamp"]))

    assert keys(tr).isdisjoint(keys(va))
    assert keys(tr).isdisjoint(keys(te))
    assert keys(va).isdisjoint(keys(te))


def test_split_files_exist(sandbox, write_small_splits):
    # sandbox fixture already chdir into tmp
    import os

    assert os.path.exists("data/processed/splits/train.parquet")
    assert os.path.exists("data/processed/splits/val.parquet")
    assert os.path.exists("data/processed/splits/test.parquet")
