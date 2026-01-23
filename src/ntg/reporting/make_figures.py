from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class Paths:
    out_dir: Path = Path("reports/figures")

    train: Path = Path("data/processed/splits/train.parquet")
    val: Path = Path("data/processed/splits/val.parquet")
    test: Path = Path("data/processed/splits/test.parquet")

    user_features: Path = Path("data/features/user_features.parquet")
    item_features: Path = Path("data/features/item_features.parquet")

    item_item: Path = Path("outputs/graph/item_item.parquet")
    clusters: Path = Path("outputs/graph/taste_clusters.parquet")

    topk: Path = Path("outputs/recommendations/topk.parquet")
    ranking_metrics: Path = Path("reports/ranking_metrics.json")

    churn_scores: Path = Path("outputs/risk/churn_scores.parquet")
    user_ltv: Path = Path("outputs/risk/user_ltv.parquet")
    user_risk: Path = Path("outputs/risk/user_risk.parquet")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_split_timeline(paths: Paths) -> None:
    if not (paths.train.exists() and paths.val.exists() and paths.test.exists()):
        return

    train = pd.read_parquet(paths.train, columns=["timestamp"], engine="pyarrow")
    val = pd.read_parquet(paths.val, columns=["timestamp"], engine="pyarrow")
    test = pd.read_parquet(paths.test, columns=["timestamp"], engine="pyarrow")

    # Convert to datetime if needed
    for df in (train, val, test):
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    fig = plt.figure()
    ax = plt.gca()

    # Bin counts by month (works even for large datasets)
    def count_by_month(df: pd.DataFrame) -> pd.Series:
        s = df["timestamp"].dropna()
        return s.dt.to_period("M").value_counts().sort_index()

    c_tr = count_by_month(train)
    c_va = count_by_month(val)
    c_te = count_by_month(test)

    ax.plot(c_tr.index.astype(str), c_tr.values, label="train")
    ax.plot(c_va.index.astype(str), c_va.values, label="val")
    ax.plot(c_te.index.astype(str), c_te.values, label="test")

    ax.set_title("Interactions Over Time (Split Timeline)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Interactions")
    ax.legend()

    # Fewer x ticks for readability
    if len(c_tr) > 24:
        for label in ax.get_xticklabels()[::2]:
            label.set_visible(False)
    ax.tick_params(axis="x", rotation=45)

    _save(fig, paths.out_dir / "split_timeline.png")


def plot_user_item_histograms(paths: Paths) -> None:
    if not paths.train.exists():
        return
    df = pd.read_parquet(paths.train, columns=["user_id", "item_id"], engine="pyarrow")

    # interactions per user
    u = df.groupby("user_id").size()
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(u.values, bins=50)
    ax.set_title("Train: Interactions per User")
    ax.set_xlabel("# interactions")
    ax.set_ylabel("# users")
    _save(fig, paths.out_dir / "hist_interactions_per_user.png")

    # interactions per item
    it = df.groupby("item_id").size()
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(it.values, bins=50)
    ax.set_title("Train: Interactions per Item")
    ax.set_xlabel("# interactions")
    ax.set_ylabel("# items")
    _save(fig, paths.out_dir / "hist_interactions_per_item.png")


def plot_cosine_distribution(paths: Paths) -> None:
    if not paths.item_item.exists():
        return
    g = pd.read_parquet(paths.item_item, columns=["cosine"], engine="pyarrow")
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(g["cosine"].values, bins=60)
    ax.set_title("Taste Graph: Cosine Similarity Distribution")
    ax.set_xlabel("cosine")
    ax.set_ylabel("# edges")
    _save(fig, paths.out_dir / "graph_cosine_distribution.png")


def plot_degree_distribution(paths: Paths) -> None:
    if not paths.item_item.exists():
        return
    g = pd.read_parquet(paths.item_item, columns=["src_item"], engine="pyarrow")
    deg = g.groupby("src_item").size()
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(deg.values, bins=50)
    ax.set_title("Taste Graph: Out-Degree Distribution (neighbors per item)")
    ax.set_xlabel("# neighbors")
    ax.set_ylabel("# items")
    _save(fig, paths.out_dir / "graph_degree_distribution.png")


def plot_cluster_sizes(paths: Paths) -> None:
    if not paths.clusters.exists():
        return
    c = pd.read_parquet(paths.clusters, engine="pyarrow")
    # expected columns: cluster_id, item_id, maybe score
    col = "cluster_id" if "cluster_id" in c.columns else ("cluster" if "cluster" in c.columns else None)
    if col is None:
        return
    sizes = c.groupby(col).size().sort_values(ascending=False).head(30)

    fig = plt.figure()
    ax = plt.gca()
    ax.bar(sizes.index.astype(str), sizes.values)
    ax.set_title("Taste Clusters: Top-30 Cluster Sizes")
    ax.set_xlabel("cluster")
    ax.set_ylabel("# items")
    ax.tick_params(axis="x", rotation=90)
    _save(fig, paths.out_dir / "clusters_top30_sizes.png")


def plot_ranker_coverage(paths: Paths) -> None:
    if not paths.topk.exists():
        return
    rec = pd.read_parquet(paths.topk, engine="pyarrow")

    # Try to infer column names
    user_col = "user_id" if "user_id" in rec.columns else None
    item_col = "item_id" if "item_id" in rec.columns else ("dst_item" if "dst_item" in rec.columns else None)
    if user_col is None or item_col is None:
        return

    n_users = rec[user_col].nunique()
    n_items = rec[item_col].nunique()

    fig = plt.figure()
    ax = plt.gca()
    ax.bar(["users", "unique recommended items"], [n_users, n_items])
    ax.set_title("Ranker: Recommendation Coverage")
    ax.set_ylabel("count")
    _save(fig, paths.out_dir / "ranker_coverage.png")


def plot_risk_quadrant(paths: Paths) -> None:
    # churn_prob vs ltv (risk quadrant)
    if not (paths.churn_scores.exists() and paths.user_ltv.exists()):
        return

    churn = pd.read_parquet(paths.churn_scores, engine="pyarrow")
    ltv = pd.read_parquet(paths.user_ltv, engine="pyarrow")

    # infer columns
    ucol = "user_id" if "user_id" in churn.columns else None
    if ucol is None:
        return

    churn_col = None
    for cand in ["churn_prob", "p_churn", "score", "churn_score"]:
        if cand in churn.columns:
            churn_col = cand
            break
    if churn_col is None:
        return

    ltv_col = None
    for cand in ["ltv_usd", "ltv", "user_ltv", "ltv_proxy"]:
        if cand in ltv.columns:
            ltv_col = cand
            break
    if ltv_col is None:
        # fallback to first numeric col
        nums = ltv.select_dtypes(include="number").columns.tolist()
        if nums:
            ltv_col = nums[0]
        else:
            return

    df = churn[[ucol, churn_col]].merge(ltv[[ucol, ltv_col]], on=ucol, how="inner")
    df = df.dropna()
    if len(df) == 0:
        return

    # downsample for plotting speed
    if len(df) > 20000:
        df = df.sample(20000, random_state=7)

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(df[churn_col].values, df[ltv_col].values, s=6)
    ax.set_title("Risk Quadrant: Churn Probability vs LTV Proxy")
    ax.set_xlabel(churn_col)
    ax.set_ylabel(ltv_col)
    _save(fig, paths.out_dir / "risk_quadrant_churn_vs_ltv.png")


def main() -> None:
    paths = Paths()
    _ensure_dir(paths.out_dir)

    plot_split_timeline(paths)
    plot_user_item_histograms(paths)
    plot_cosine_distribution(paths)
    plot_degree_distribution(paths)
    plot_cluster_sizes(paths)
    plot_ranker_coverage(paths)
    plot_risk_quadrant(paths)

    print(f"✅ Wrote figures to: {paths.out_dir}")


if __name__ == "__main__":
    main()
