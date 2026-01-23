# src/ntg/reporting/metrics_summary.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# ============================================================
# Metrics Summary (Recruiter-Friendly)
# - Reads only produced artifacts (parquet/json)
# - Avoids scanning full large parquets (reads minimal columns)
# - Never crashes if a file is missing
# - Prints markdown table for README copy/paste
# ============================================================

@dataclass(frozen=True)
class Paths:
    splits_meta: Path = Path("data/processed/splits/metadata.json")
    train: Path = Path("data/processed/splits/train.parquet")
    val: Path = Path("data/processed/splits/val.parquet")
    test: Path = Path("data/processed/splits/test.parquet")

    user_features: Path = Path("data/features/user_features.parquet")
    item_features: Path = Path("data/features/item_features.parquet")
    interaction_features: Path = Path("data/features/interaction_features_train.parquet")

    item_item: Path = Path("outputs/graph/item_item.parquet")
    clusters: Path = Path("outputs/graph/taste_clusters.parquet")
    item_item_meta: Path = Path("outputs/graph/item_item_meta.json")
    clusters_meta: Path = Path("outputs/graph/taste_clusters_meta.json")

    topk: Path = Path("outputs/recommendations/topk.parquet")
    ranking_metrics: Path = Path("reports/ranking_metrics.json")

    churn_scores: Path = Path("outputs/risk/churn_scores.parquet")
    churn_report: Path = Path("reports/churn_model_report.json")
    user_ltv: Path = Path("outputs/risk/user_ltv.parquet")
    revenue_summary: Path = Path("reports/revenue_risk_summary.json")

    ab_3arm: Path = Path("reports/ab_results_3arm.json")


# ----------------------------
# helpers
# ----------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_parquet_shape(path: Path) -> Optional[Tuple[int, int]]:
    if not path.exists():
        return None
    # This does read metadata + schema quickly; still OK.
    # For huge parquets, pyarrow reads footer metadata without scanning rows.
    df = pd.read_parquet(path, engine="pyarrow")
    return df.shape


def _detect_cols(cols: set[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    user_col = "user_id" if "user_id" in cols else None
    item_col = "item_id" if "item_id" in cols else None
    ts_col = "timestamp" if "timestamp" in cols else ("ts" if "ts" in cols else None)
    return user_col, item_col, ts_col


def _split_stats(split_path: Path) -> Dict[str, Optional[Any]]:
    if not split_path.exists():
        return {"rows": None, "users": None, "items": None, "t_min": None, "t_max": None}

    # Read only necessary columns (fast + safe)
    # First peek columns from minimal read
    df0 = pd.read_parquet(split_path, engine="pyarrow")
    cols = set(df0.columns)
    user_col, item_col, ts_col = _detect_cols(cols)

    needed = [c for c in [user_col, item_col, ts_col] if c is not None]
    if needed:
        df = pd.read_parquet(split_path, columns=needed, engine="pyarrow")
    else:
        df = df0

    users = int(df[user_col].nunique(dropna=True)) if user_col else None
    items = int(df[item_col].nunique(dropna=True)) if item_col else None
    t_min = str(df[ts_col].min()) if ts_col else None
    t_max = str(df[ts_col].max()) if ts_col else None
    return {"rows": int(len(df0)), "users": users, "items": items, "t_min": t_min, "t_max": t_max}


def _safe_unique(path: Path, col: str) -> Optional[int]:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=[col], engine="pyarrow")
    return int(df[col].nunique(dropna=True))


def _fmt3(x: Any) -> Any:
    # small formatter to keep table readable
    if isinstance(x, float):
        return float(f"{x:.6g}")
    return x


def build_summary_table(paths: Paths = Paths()) -> pd.DataFrame:
    splits_meta = _read_json(paths.splits_meta)

    train_stats = _split_stats(paths.train)
    val_stats = _split_stats(paths.val)
    test_stats = _split_stats(paths.test)

    uf_shape = _safe_parquet_shape(paths.user_features)
    it_shape = _safe_parquet_shape(paths.item_features)
    ix_shape = _safe_parquet_shape(paths.interaction_features)

    graph_shape = _safe_parquet_shape(paths.item_item)
    graph_items = _safe_unique(paths.item_item, "src_item")
    clusters_shape = _safe_parquet_shape(paths.clusters)

    item_meta = _read_json(paths.item_item_meta)
    # robust fallback keys
    n_items_supported = item_meta.get("n_items_supported") or item_meta.get("n_items_retained")
    n_edges_topk = item_meta.get("n_edges_directed_topk") or item_meta.get("n_edges") or item_meta.get("n_edges_topk")

    rank_shape = _safe_parquet_shape(paths.topk)
    rank_json = _read_json(paths.ranking_metrics)

    churn_shape = _safe_parquet_shape(paths.churn_scores)
    churn_json = _read_json(paths.churn_report)

    ltv_shape = _safe_parquet_shape(paths.user_ltv)
    rev_json = _read_json(paths.revenue_summary)

    ab_json = _read_json(paths.ab_3arm)

    rows: list[tuple[str, str, Any]] = []

    # Day-1
    rows.append(("Day-1 Splits", "strategy", splits_meta.get("strategy")))
    rows.append(("Day-1 Splits", "n_total", splits_meta.get("n_total")))
    rows.append(
        ("Day-1 Splits", "train/val/test frac",
         f"{_fmt3(splits_meta.get('frac_train'))} / {_fmt3(splits_meta.get('frac_val'))} / {_fmt3(splits_meta.get('frac_test'))}")
    )
    rows.append(("Day-1 Splits", "train users/items", f"{train_stats['users']} / {train_stats['items']}"))
    rows.append(("Day-1 Splits", "train time range", f"{train_stats['t_min']} → {train_stats['t_max']}"))

    # Day-2
    rows.append(("Day-2 Features", "user_features shape", uf_shape))
    rows.append(("Day-2 Features", "item_features shape", it_shape))
    rows.append(("Day-2 Features", "interaction_features shape", ix_shape))

    # Day-3
    rows.append(("Day-3 Graph", "item-item edges shape", graph_shape))
    rows.append(("Day-3 Graph", "items with neighbors", graph_items))
    rows.append(("Day-3 Graph", "n_items_supported", n_items_supported))
    rows.append(("Day-3 Graph", "n_edges_directed_topk", n_edges_topk))
    rows.append(("Day-3 Graph", "clusters shape", clusters_shape))

    # Day-4/5
    rows.append(("Day-4/5 Ranker", "topk parquet shape", rank_shape))
    for k in ["recall_at_k", "ndcg_at_k", "map_at_k", "precision_at_k", "k"]:
        if k in rank_json:
            rows.append(("Day-4/5 Ranker", k, _fmt3(rank_json.get(k))))

    # Day-6/7
    rows.append(("Day-6/7 Risk", "churn_scores shape", churn_shape))
    for k in ["auc", "accuracy", "brier", "n_users", "positive_rate"]:
        if k in churn_json:
            rows.append(("Day-6/7 Risk", f"churn_{k}", _fmt3(churn_json.get(k))))
    rows.append(("Day-6/7 Risk", "user_ltv shape", ltv_shape))
    for k in ["total_revenue_at_risk_usd", "top_decile_revenue_at_risk_usd", "n_users_scored"]:
        if k in rev_json:
            rows.append(("Day-6/7 Risk", k, _fmt3(rev_json.get(k))))

    # Day-10/11
    if ab_json:
        rows.append(("Day-10/11 A/B/n", "experiment_id", ab_json.get("experiment_id")))
        rows.append(("Day-10/11 A/B/n", "n_total", ab_json.get("n_total")))
        if ab_json.get("primary_metric"):
            rows.append(("Day-10/11 A/B/n", "primary_metric", ab_json.get("primary_metric")))

    # Build DF + clean display
    df = pd.DataFrame(rows, columns=["Phase", "Metric", "Value"])
    df["Value"] = df["Value"].apply(lambda x: "" if x is None else x)
    return df


def main() -> None:
    df = build_summary_table()
    try:
        # Requires `tabulate` installed for pandas markdown
        print(df.to_markdown(index=False))
    except Exception:
        # fallback, always works
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
