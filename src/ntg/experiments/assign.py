from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import pandas as pd


Variant = Literal["control", "treatment"]


@dataclass(frozen=True)
class AssignmentConfig:
    experiment_id: str = "ntg_ab_v1"
    # 0.5 means 50/50 split
    treatment_share: float = 0.50


def _stable_hash_to_unit_interval(experiment_id: str, user_id: int) -> float:
    """
    Deterministic hash -> [0,1).
    This makes assignments stable across reruns and independent of ordering.
    """
    key = f"{experiment_id}:{int(user_id)}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    # use first 16 hex chars to make a 64-bit int
    x = int(h[:16], 16)
    return (x % (10**12)) / (10**12)


def assign_variants(df: pd.DataFrame, cfg: AssignmentConfig) -> pd.DataFrame:
    """
    Adds column: variant in {"control","treatment"}.
    """
    if "user_id" not in df.columns:
        raise ValueError("assign_variants expects a user_id column")

    if not (0.0 < cfg.treatment_share < 1.0):
        raise ValueError("treatment_share must be in (0,1)")

    out = df.copy()
    u = out["user_id"].astype("int64").to_numpy()

    r = [_stable_hash_to_unit_interval(cfg.experiment_id, int(uid)) for uid in u]
    out["variant"] = ["treatment" if x < cfg.treatment_share else "control" for x in r]
    return out
