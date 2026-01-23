# src/ntg/data/splits.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


# -------------------------------
# Configs (backward compatible)
# -------------------------------
@dataclass(frozen=True)
class TimeSplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    min_interactions: int = 5


# Older names some tests/code may look for
PerUserTimeSplitConfig = TimeSplitConfig
GlobalTimeSplitConfig = TimeSplitConfig


def _validate_cfg(cfg: TimeSplitConfig) -> None:
    if cfg.train_frac <= 0 or cfg.train_frac >= 1:
        raise ValueError("train_frac must be in (0, 1)")
    if cfg.val_frac < 0 or cfg.val_frac >= 1:
        raise ValueError("val_frac must be in [0, 1)")
    if cfg.train_frac + cfg.val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")
    if cfg.min_interactions < 1:
        raise ValueError("min_interactions must be >= 1")


def _require_columns(df: pd.DataFrame) -> None:
    required = {"user_id", "item_id", "timestamp", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dtypes so tests behave deterministically and filters like df[df["user_id"]==2]
    work as expected (no string/int mismatches).
    """
    out = df.copy()

    # user_id / item_id must be numeric ints
    out["user_id"] = pd.to_numeric(out["user_id"], errors="raise").astype("int64")
    out["item_id"] = pd.to_numeric(out["item_id"], errors="raise").astype("int64")

    # rating numeric float
    out["rating"] = pd.to_numeric(out["rating"], errors="raise").astype("float64")

    # timestamp datetime
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("timestamp contains NaT after conversion")

    return out


# ==========================================================
# Global chronological split
# ==========================================================
def time_split_global(
    interactions: pd.DataFrame,
    cfg: TimeSplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    _validate_cfg(cfg)
    _require_columns(interactions)

    if interactions.empty:
        meta = {
            "strategy": "global_time",
            "n_total": 0,
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
        }
        empty = interactions.copy()
        return empty, empty, empty, meta

    df = _coerce_types(interactions)
    df = df.sort_values(["timestamp", "user_id", "item_id"], ascending=[True, True, True]).reset_index(drop=True)

    n = len(df)
    train_end = int(n * float(cfg.train_frac))  # floor
    val_end = int(n * float(cfg.train_frac + cfg.val_frac))  # floor

    train_end = max(1, train_end)
    val_end = max(train_end, val_end)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    if len(train) + len(val) + len(test) != n:
        raise RuntimeError("Global split produced row loss/gain (train+val+test != total)")

    meta = {
        "strategy": "global_time",
        "n_total": int(n),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "frac_train": float(len(train) / max(n, 1)),
        "frac_val": float(len(val) / max(n, 1)),
        "frac_test": float(len(test) / max(n, 1)),
        "train_frac": float(cfg.train_frac),
        "val_frac": float(cfg.val_frac),
    }
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), meta


# ==========================================================
# Per-user chronological split
# ==========================================================
def time_split_per_user(
    interactions: pd.DataFrame,
    cfg: TimeSplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    _validate_cfg(cfg)
    _require_columns(interactions)

    if interactions.empty:
        meta = {
            "strategy": "per_user_time",
            "n_total": 0,
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
            "min_interactions": cfg.min_interactions,
        }
        empty = interactions.copy()
        return empty, empty, empty, meta

    df = _coerce_types(interactions)

    df = df.sort_values(["user_id", "timestamp", "item_id"], ascending=[True, True, True]).reset_index(drop=True)

    df["_cnt"] = df.groupby("user_id")["user_id"].transform("size").astype("int64")
    df["_rn"] = (df.groupby("user_id").cumcount() + 1).astype("int64")

    train_end = (df["_cnt"] * float(cfg.train_frac)).apply(lambda x: int(x)).clip(lower=1).astype("int64")
    val_end = (df["_cnt"] * float(cfg.train_frac + cfg.val_frac)).apply(lambda x: int(x)).astype("int64")
    val_end = pd.concat([train_end, val_end], axis=1).max(axis=1).astype("int64")

    df["_train_end"] = train_end
    df["_val_end"] = val_end

    # Assign split (min_interactions override)
    def _assign_split(row) -> str:
        if int(row["_cnt"]) < int(cfg.min_interactions):
            return "train"
        if int(row["_rn"]) <= int(row["_train_end"]):
            return "train"
        if int(row["_rn"]) <= int(row["_val_end"]):
            return "val"
        return "test"

    df["split"] = df.apply(_assign_split, axis=1)

    train = df[df["split"] == "train"].drop(columns=["_cnt", "_rn", "_train_end", "_val_end", "split"])
    val = df[df["split"] == "val"].drop(columns=["_cnt", "_rn", "_train_end", "_val_end", "split"])
    test = df[df["split"] == "test"].drop(columns=["_cnt", "_rn", "_train_end", "_val_end", "split"])

    n_total = len(df)
    if len(train) + len(val) + len(test) != n_total:
        raise RuntimeError("Per-user split produced row loss/gain (train+val+test != total)")

    meta = {
        "strategy": "per_user_time",
        "n_total": int(n_total),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "frac_train": float(len(train) / max(n_total, 1)),
        "frac_val": float(len(val) / max(n_total, 1)),
        "frac_test": float(len(test) / max(n_total, 1)),
        "train_frac": float(cfg.train_frac),
        "val_frac": float(cfg.val_frac),
        "min_interactions": int(cfg.min_interactions),
    }

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), meta


# Backward-compat alias some code may use
time_split_config_per_user = time_split_per_user


def write_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    metadata: Dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    if metadata is not None:
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
