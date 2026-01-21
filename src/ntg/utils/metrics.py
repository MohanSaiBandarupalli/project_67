from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingMetricsConfig:
    """
    Offline ranking metrics config.

    k_list: evaluate metrics at these cutoffs
    """
    k_list: Tuple[int, ...] = (5, 10, 20, 50)


def _dcg(rels: np.ndarray) -> float:
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def _ndcg_at_k(rec_items: Sequence[int], gt_set: set[int], k: int) -> float:
    if k <= 0 or len(gt_set) == 0:
        return 0.0
    ranked = rec_items[:k]
    rels = np.array([1.0 if x in gt_set else 0.0 for x in ranked], dtype=np.float64)
    dcg = _dcg(rels)
    ideal_len = min(k, len(gt_set))
    idcg = _dcg(np.ones(ideal_len, dtype=np.float64))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _ap_at_k(rec_items: Sequence[int], gt_set: set[int], k: int) -> float:
    if k <= 0 or len(gt_set) == 0:
        return 0.0
    ranked = rec_items[:k]
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(ranked, start=1):
        if item in gt_set:
            hits += 1
            sum_prec += hits / i
    denom = min(len(gt_set), k)
    return float(sum_prec / denom) if denom > 0 else 0.0


def _precision_recall_at_k(rec_items: Sequence[int], gt_set: set[int], k: int) -> Tuple[float, float]:
    if k <= 0:
        return 0.0, 0.0
    ranked = rec_items[:k]
    hit = sum(1 for x in ranked if x in gt_set)
    precision = hit / k
    recall = hit / max(len(gt_set), 1)
    return float(precision), float(recall)


def _hit_rate_at_k(rec_items: Sequence[int], gt_set: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    ranked = rec_items[:k]
    return 1.0 if any(x in gt_set for x in ranked) else 0.0


def evaluate_topk(
    topk: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cfg: RankingMetricsConfig = RankingMetricsConfig(),
) -> Dict[str, float]:
    """
    Evaluate macro-averaged offline ranking metrics.

    Required topk columns:
      - user_id, item_id
      - rank OR score

    Required ground_truth columns:
      - user_id, item_id
    """
    if not {"user_id", "item_id"}.issubset(topk.columns):
        raise ValueError("topk must have columns: user_id, item_id")
    if not {"user_id", "item_id"}.issubset(ground_truth.columns):
        raise ValueError("ground_truth must have columns: user_id, item_id")

    df = topk.copy()
    gt = ground_truth.copy()

    # Ensure deterministic per-user ordering
    if "rank" in df.columns:
        df = df.sort_values(["user_id", "rank", "item_id"], ascending=[True, True, True])
    elif "score" in df.columns:
        df = df.sort_values(["user_id", "score", "item_id"], ascending=[True, False, True])
        df["rank"] = df.groupby("user_id").cumcount() + 1
    else:
        raise ValueError("topk must include either 'rank' or 'score'")

    gt_sets: Dict[int, set[int]] = (
        gt.groupby("user_id")["item_id"]
        .apply(lambda s: set(map(int, s.tolist())))
        .to_dict()
    )

    rec_lists: Dict[int, list[int]] = (
        df.groupby("user_id")["item_id"]
        .apply(lambda s: list(map(int, s.tolist())))
        .to_dict()
    )

    users = [u for u in rec_lists.keys() if u in gt_sets and len(gt_sets[u]) > 0]
    if not users:
        return {"n_users": 0.0}

    out: Dict[str, float] = {"n_users": float(len(users))}
    for k in cfg.k_list:
        precs, recs, ndcgs, maps, hits = [], [], [], [], []
        for u in users:
            gt_set = gt_sets[u]
            rec_items = rec_lists[u]
            p, r = _precision_recall_at_k(rec_items, gt_set, k)
            precs.append(p)
            recs.append(r)
            ndcgs.append(_ndcg_at_k(rec_items, gt_set, k))
            maps.append(_ap_at_k(rec_items, gt_set, k))
            hits.append(_hit_rate_at_k(rec_items, gt_set, k))

        out[f"precision@{k}"] = float(np.mean(precs))
        out[f"recall@{k}"] = float(np.mean(recs))
        out[f"ndcg@{k}"] = float(np.mean(ndcgs))
        out[f"map@{k}"] = float(np.mean(maps))
        out[f"hitrate@{k}"] = float(np.mean(hits))

    # simple coverage numbers
    out["unique_items_recommended"] = float(df["item_id"].nunique())
    out["total_recommendations"] = float(len(df))
    out["item_coverage_ratio"] = float(df["item_id"].nunique() / max(len(df), 1))
    return out
