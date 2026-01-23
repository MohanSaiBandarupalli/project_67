# src/ntg/evaluation/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingMetricsConfig:
    """Config for offline top-k evaluation."""
    k_list: Tuple[int, ...] = (5, 10, 20, 50)
    # If your GT can contain duplicates, we treat it as set per user.
    dedup_ground_truth: bool = True


def _require_cols(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Found: {list(df.columns)}")


def _dcg(rels: np.ndarray) -> float:
    # rels is binary relevance vector in ranked order
    if rels.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rels.size + 2))
    return float((rels / denom).sum())


def evaluate_topk(
    topk: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cfg: RankingMetricsConfig | None = None,
) -> Dict[str, float]:
    """
    Evaluate top-k recommendations against ground truth interactions.
    Required columns:
      - topk: user_id, item_id, rank  (rank 1 is best)  [score optional]
      - ground_truth: user_id, item_id

    Returns a flat dict with metrics for each k in cfg.k_list.
    """
    if cfg is None:
        cfg = RankingMetricsConfig()

    # Validate inputs
    if topk is None or ground_truth is None:
        return {"warning": "topk or ground_truth is None"}

    if len(topk) == 0 or len(ground_truth) == 0:
        # Still return keys for stability
        out: Dict[str, float] = {"n_users_eval": 0.0}
        for k in cfg.k_list:
            out[f"precision@{k}"] = 0.0
            out[f"recall@{k}"] = 0.0
            out[f"hit_rate@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
            out[f"map@{k}"] = 0.0
        return out

    _require_cols(topk, ["user_id", "item_id"], "topk")
    _require_cols(ground_truth, ["user_id", "item_id"], "ground_truth")

    # rank is optional but recommended; if absent, assume current order is ranked
    if "rank" in topk.columns:
        recs = topk.sort_values(["user_id", "rank", "item_id"], ascending=[True, True, True]).copy()
    else:
        recs = topk.sort_values(["user_id", "item_id"], ascending=[True, True]).copy()
        recs["rank"] = recs.groupby("user_id").cumcount() + 1

    gt = ground_truth[["user_id", "item_id"]].copy()
    if cfg.dedup_ground_truth:
        gt = gt.drop_duplicates(["user_id", "item_id"])

    # Build per-user GT sets
    gt_sets = gt.groupby("user_id")["item_id"].apply(lambda s: set(s.tolist()))
    # Evaluate only users that exist in both
    users = np.intersect1d(recs["user_id"].unique(), gt_sets.index.values)

    if users.size == 0:
        out: Dict[str, float] = {"n_users_eval": 0.0}
        for k in cfg.k_list:
            out[f"precision@{k}"] = 0.0
            out[f"recall@{k}"] = 0.0
            out[f"hit_rate@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
            out[f"map@{k}"] = 0.0
        return out

    recs = recs[recs["user_id"].isin(users)].copy()

    out: Dict[str, float] = {"n_users_eval": float(users.size)}

    # Pre-split recs per user for speed and determinism
    recs_by_user = {
        uid: grp.sort_values("rank", ascending=True)["item_id"].astype("int64", errors="ignore").tolist()
        for uid, grp in recs.groupby("user_id", sort=True)
    }

    for k in cfg.k_list:
        precisions = []
        recalls = []
        hits = []
        ndcgs = []
        maps = []

        for uid in users:
            gt_u = gt_sets.loc[uid]
            if not gt_u:
                continue

            rec_u = recs_by_user.get(uid, [])
            rec_k = rec_u[:k]

            if not rec_k:
                precisions.append(0.0)
                recalls.append(0.0)
                hits.append(0.0)
                ndcgs.append(0.0)
                maps.append(0.0)
                continue

            hit_flags = np.array([1.0 if it in gt_u else 0.0 for it in rec_k], dtype=float)
            n_hits = float(hit_flags.sum())

            precision = n_hits / float(len(rec_k))
            recall = n_hits / float(len(gt_u))
            hit_rate = 1.0 if n_hits > 0.0 else 0.0

            # NDCG
            dcg = _dcg(hit_flags)
            ideal = np.ones(int(min(len(gt_u), k)), dtype=float)
            idcg = _dcg(ideal)
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            # MAP@k (binary relevance)
            if n_hits == 0.0:
                ap = 0.0
            else:
                cum_hits = np.cumsum(hit_flags)
                prec_at_i = cum_hits / (np.arange(len(hit_flags)) + 1.0)
                ap = float((prec_at_i * hit_flags).sum() / min(len(gt_u), k))

            precisions.append(precision)
            recalls.append(recall)
            hits.append(hit_rate)
            ndcgs.append(ndcg)
            maps.append(ap)

        # If all users had empty GT (rare), keep zeros
        out[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        out[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        out[f"hit_rate@{k}"] = float(np.mean(hits)) if hits else 0.0
        out[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        out[f"map@{k}"] = float(np.mean(maps)) if maps else 0.0

    return out
