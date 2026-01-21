from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import pandas as pd


Variant2 = Literal["control", "treatment"]
Variant3 = Literal["control", "treatment_content", "treatment_discount"]


@dataclass(frozen=True)
class AssignmentConfig:
    experiment_id: str = "ntg_ab_v1"
    treatment_share: float = 0.50


@dataclass(frozen=True)
class Assignment3Config:
    experiment_id: str = "ntg_ab_v2_3arm"
    share_content: float = 0.45
    share_discount: float = 0.05  # small, expensive arm
    # control share is implied: 1 - share_content - share_discount


def _stable_hash_to_unit_interval(experiment_id: str, user_id: int) -> float:
    key = f"{experiment_id}:{int(user_id)}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    x = int(h[:16], 16)
    return (x % (10**12)) / (10**12)


def assign_variants(df: pd.DataFrame, cfg: AssignmentConfig) -> pd.DataFrame:
    if "user_id" not in df.columns:
        raise ValueError("assign_variants expects a user_id column")
    if not (0.0 < cfg.treatment_share < 1.0):
        raise ValueError("treatment_share must be in (0,1)")

    out = df.copy()
    u = out["user_id"].astype("int64").to_numpy()
    r = [_stable_hash_to_unit_interval(cfg.experiment_id, int(uid)) for uid in u]
    out["variant"] = ["treatment" if x < cfg.treatment_share else "control" for x in r]
    return out


def assign_variants_3arm(df: pd.DataFrame, cfg: Assignment3Config) -> pd.DataFrame:
    if "user_id" not in df.columns:
        raise ValueError("assign_variants_3arm expects a user_id column")

    if cfg.share_content < 0 or cfg.share_discount < 0:
        raise ValueError("shares must be >= 0")
    if cfg.share_content + cfg.share_discount >= 1.0:
        raise ValueError("share_content + share_discount must be < 1")

    out = df.copy()
    u = out["user_id"].astype("int64").to_numpy()

    r = [_stable_hash_to_unit_interval(cfg.experiment_id, int(uid)) for uid in u]
    c = cfg.share_content
    d = cfg.share_discount

    variant: list[str] = []
    for x in r:
        if x < d:
            variant.append("treatment_discount")
        elif x < d + c:
            variant.append("treatment_content")
        else:
            variant.append("control")

    out["variant"] = variant
    return out
