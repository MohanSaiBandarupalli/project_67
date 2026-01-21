# src/ntg/data/splits.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from ntg.data.schemas import SCHEMA


# -------------------------
# Per-user chronological split
# -------------------------

@dataclass(frozen=True)
class PerUserTimeSplitConfig:
    """
    Per-user chronological split config.

    train_frac + val_frac must be < 1.0.
    test_frac is implied.

    min_interactions:
      - if a user has fewer than this, keep all rows in train
        to avoid pathological tiny val/test for that user.
    """
    train_frac: float = 0.70
    val_frac: float = 0.15
    min_interactions: int = 5

    def validate(self) -> None:
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        if not (0.0 <= self.val_frac < 1.0):
            raise ValueError("val_frac must be in [0, 1)")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")
        if self.min_interactions < 1:
            raise ValueError("min_interactions must be >= 1")


def _split_one_user(g: pd.DataFrame, cfg: PerUserTimeSplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # g is already sorted by timestamp
    n = len(g)
    if n < cfg.min_interactions:
        return g, g.iloc[0:0], g.iloc[0:0]

    train_end = max(int(n * cfg.train_frac), 1)
    val_end = max(int(n * (cfg.train_frac + cfg.val_frac)), train_end)

    train = g.iloc[:train_end]
    val = g.iloc[train_end:val_end]
    test = g.iloc[val_end:]
    return train, val, test


def time_split_per_user(
    df: pd.DataFrame,
    cfg: PerUserTimeSplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Returns (train, val, test, metadata)

    Guarantees:
      - within each user, train rows are earlier than val, earlier than test
      - deterministic (no randomization)
    """
    cfg.validate()

    df = df.sort_values([SCHEMA.USER_ID, SCHEMA.TIMESTAMP, SCHEMA.ITEM_ID]).reset_index(drop=True)

    train_parts, val_parts, test_parts = [], [], []

    for _, g in df.groupby(SCHEMA.USER_ID, sort=False):
        tr, va, te = _split_one_user(g, cfg)
        train_parts.append(tr)
        if len(va) > 0:
            val_parts.append(va)
        if len(te) > 0:
            test_parts.append(te)

    train = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    val = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0]

    meta = {
        "n_total": float(len(df)),
        "n_train": float(len(train)),
        "n_val": float(len(val)),
        "n_test": float(len(test)),
        "frac_train": float(len(train) / max(len(df), 1)),
        "frac_val": float(len(val) / max(len(df), 1)),
        "frac_test": float(len(test) / max(len(df), 1)),
        "min_interactions": float(cfg.min_interactions),
        "train_frac": float(cfg.train_frac),
        "val_frac": float(cfg.val_frac),
        "strategy": "per_user_time",
    }
    return train, val, test, meta


# -------------------------
# Global time split (scales better)
# -------------------------

@dataclass(frozen=True)
class GlobalTimeSplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15

    def validate(self) -> None:
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        if not (0.0 <= self.val_frac < 1.0):
            raise ValueError("val_frac must be in [0, 1)")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")


def time_split_global(
    df: pd.DataFrame,
    cfg: GlobalTimeSplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Split by global timestamp order.

    Pros:
      - simpler
      - strong global leakage protection

    Cons:
      - not per-user aligned
    """
    cfg.validate()
    df = df.sort_values([SCHEMA.TIMESTAMP, SCHEMA.USER_ID, SCHEMA.ITEM_ID]).reset_index(drop=True)

    n = len(df)
    t_end = int(n * cfg.train_frac)
    v_end = int(n * (cfg.train_frac + cfg.val_frac))

    train = df.iloc[:t_end]
    val = df.iloc[t_end:v_end]
    test = df.iloc[v_end:]

    meta = {
        "n_total": float(n),
        "n_train": float(len(train)),
        "n_val": float(len(val)),
        "n_test": float(len(test)),
        "frac_train": float(len(train) / max(n, 1)),
        "frac_val": float(len(val) / max(n, 1)),
        "frac_test": float(len(test) / max(n, 1)),
        "train_frac": float(cfg.train_frac),
        "val_frac": float(cfg.val_frac),
        "strategy": "global_time",
    }
    return train, val, test, meta
