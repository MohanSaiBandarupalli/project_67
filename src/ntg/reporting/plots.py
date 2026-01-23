# src/ntg/reporting/plots.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import math
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _save(fig: plt.Figure, out: Path, also_pdf: bool = False, dpi: int = 170) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def _maybe_log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


# -------------------------
# Paths
# -------------------------

@dataclass(frozen=True)
class FigurePaths:
    out_dir: Path = Path("reports/figures")

    splits_meta: Path = Path("data/processed/splits/metadata.json")
    train: Path = Path("data/processed/splits/train.parquet")

    item_item: Path = Path("outputs/graph/item_item.parquet")
    item_item_meta: Path = Path("outputs/graph/item_item_meta.json")
    clusters: Path = Path("outputs/graph/taste_clusters.parquet")

    topk: Path = Path("outputs/recommendations/topk.parquet")
    ranking_metrics: Path = Path("reports/ranking_metrics.json")

    churn_scores: Path = Path("outputs/risk/churn_scores.parquet")
    user_ltv: Path = Path("outputs/risk/user_ltv.parquet")
    user_risk: Path = Path("outputs/risk/user_risk.parquet")  # if present
    revenue_summary: Path = Path("reports/revenue_risk_summary.json")

    ab_results_3arm: Path = Path("reports/ab_results_3arm.json")


# -------------------------
# A) Risk quadrant + top-decile overlay
# -------------------------

def fig_risk_quadrant(
    paths: FigurePaths,
    also_pdf: bool = False,
    max_points: int = 200_000,
    verbose: bool = False,
) -> Optional[Path]:
    if not paths.churn_scores.exists() or not paths.user_ltv.exists():
        _maybe_log("[risk_quadrant] missing churn_scores or user_ltv", verbose)
        return None

    churn = pd.read_parquet(paths.churn_scores, engine="pyarrow")
    ltv = pd.read_parquet(paths.user_ltv, engine="pyarrow")

    # Expected schema: churn_scores has user_id + churn_prob
    # user_ltv has user_id + ltv_usd + revenue_at_risk_usd (or similar)
    churn_cols = set(churn.columns)
    ltv_cols = set(ltv.columns)

    churn_prob_col = "churn_prob" if "churn_prob" in churn_cols else ("score" if "score" in churn_cols else None)
    if churn_prob_col is None:
        _maybe_log("[risk_quadrant] churn prob column not found", verbose)
        return None

    # pick columns robustly
    ltv_col = "ltv_usd" if "ltv_usd" in ltv_cols else ("ltv" if "ltv" in ltv_cols else None)
    rar_col = "revenue_at_risk_usd" if "revenue_at_risk_usd" in ltv_cols else ("rev_at_risk_usd" if "rev_at_risk_usd" in ltv_cols else None)

    if ltv_col is None:
        # fallback: pick first numeric besides user_id
        cand = [c for c in ltv.columns if c != "user_id" and pd.api.types.is_numeric_dtype(ltv[c])]
        ltv_col = cand[0] if cand else None
    if rar_col is None:
        cand = [c for c in ltv.columns if c != "user_id" and pd.api.types.is_numeric_dtype(ltv[c])]
        rar_col = cand[1] if len(cand) > 1 else None

    df = churn[["user_id", churn_prob_col]].merge(
        ltv[["user_id", ltv_col] + ([rar_col] if rar_col else [])],
        on="user_id",
        how="inner",
    )
    df = df.rename(columns={churn_prob_col: "churn_prob", ltv_col: "ltv_usd"})
    if rar_col:
        df = df.rename(columns={rar_col: "revenue_at_risk_usd"})

    # sample for speed/plot clarity
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=7)

    # define quadrants using medians
    x = df["churn_prob"].clip(0, 1)
    y = df["ltv_usd"]
    x_med = float(x.median())
    y_med = float(y.median())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x, y, s=4, alpha=0.25)

    # median lines
    ax.axvline(x_med, linestyle="--")
    ax.axhline(y_med, linestyle="--")

    ax.set_title("User Risk Quadrant: churn probability vs LTV")
    ax.set_xlabel("Churn probability (proxy)")
    ax.set_ylabel("LTV (proxy, USD)")

    # annotate quadrants
    ax.text(0.02, 0.98, "High LTV\nLow churn", transform=ax.transAxes, va="top")
    ax.text(0.72, 0.98, "High LTV\nHigh churn", transform=ax.transAxes, va="top")
    ax.text(0.02, 0.10, "Low LTV\nLow churn", transform=ax.transAxes, va="bottom")
    ax.text(0.72, 0.10, "Low LTV\nHigh churn", transform=ax.transAxes, va="bottom")

    # top-decile overlay (by revenue_at_risk if present)
    if "revenue_at_risk_usd" in df.columns:
        rar = df["revenue_at_risk_usd"]
        thr = float(rar.quantile(0.90))
        top = df[rar >= thr]
        ax.scatter(top["churn_prob"], top["ltv_usd"], s=8, alpha=0.8)
        ax.set_title("User Risk Quadrant + Top-Decile Revenue-at-Risk Overlay")

    out = paths.out_dir / "A_risk_quadrant.png"
    _save(fig, out, also_pdf=also_pdf)
    return out


# -------------------------
# B) Ranker metrics curve (K vs metric)
# -------------------------

def fig_ranker_metrics_curves(
    paths: FigurePaths,
    also_pdf: bool = False,
    ks: Sequence[int] = (5, 10, 20, 50, 100),
    verbose: bool = False,
) -> Optional[Path]:
    if not paths.ranking_metrics.exists():
        _maybe_log("[ranker_curves] missing reports/ranking_metrics.json", verbose)
        return None

    m = _read_json(paths.ranking_metrics)

    # Support both scalar and dict-per-K formats.
    # If your JSON stores e.g. {"recall_at_k": {"5":0.1,"10":0.2}} we plot curves.
    # If scalar only, we plot bars at K=reported_k or just show single point.
    metrics = ["recall_at_k", "ndcg_at_k", "map_at_k", "precision_at_k"]
    found_any = False

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for key in metrics:
        if key not in m:
            continue
        val = m[key]
        if isinstance(val, dict):
            # keys might be strings
            xs, ys = [], []
            for k in ks:
                kk = str(k)
                if kk in val:
                    xs.append(k)
                    ys.append(float(val[kk]))
            if xs:
                ax.plot(xs, ys, marker="o", label=key)
                found_any = True
        elif isinstance(val, (int, float)):
            # single value — plot as a single point at max K (or 10)
            ax.plot([max(ks)], [float(val)], marker="o", label=key)
            found_any = True

    if not found_any:
        _maybe_log("[ranker_curves] no plottable metrics found in ranking_metrics.json", verbose)
        plt.close(fig)
        return None

    ax.set_title("Ranker Quality Curves (higher is better)")
    ax.set_xlabel("K")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out = paths.out_dir / "B_ranker_metrics_curves.png"
    _save(fig, out, also_pdf=also_pdf)
    return out


# -------------------------
# C) Graph diagnostics
#    - Degree distribution (log-y)
#    - Cosine vs cooc scatter
# -------------------------

def fig_graph_degree_distribution(
    paths: FigurePaths,
    also_pdf: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    if not paths.item_item.exists():
        _maybe_log("[graph_degree] missing outputs/graph/item_item.parquet", verbose)
        return None

    g = pd.read_parquet(paths.item_item, engine="pyarrow")

    # expected columns: src_item, dst_item, cosine, cooc_cnt, ...
    if "src_item" not in g.columns:
        _maybe_log("[graph_degree] src_item missing", verbose)
        return None

    deg = g.groupby("src_item").size().astype(int)
    deg_vals = deg.values

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # histogram
    ax.hist(deg_vals, bins=60)
    ax.set_yscale("log")
    ax.set_title("Item-Item Graph: Out-degree distribution (log scale)")
    ax.set_xlabel("Out-degree (neighbors per item)")
    ax.set_ylabel("Count of items (log)")
    ax.grid(True, alpha=0.25)

    out = paths.out_dir / "C1_graph_degree_distribution.png"
    _save(fig, out, also_pdf=also_pdf)
    return out


def fig_graph_cosine_vs_cooc(
    paths: FigurePaths,
    also_pdf: bool = False,
    max_points: int = 250_000,
    verbose: bool = False,
) -> Optional[Path]:
    if not paths.item_item.exists():
        _maybe_log("[cosine_vs_cooc] missing outputs/graph/item_item.parquet", verbose)
        return None

    g = pd.read_parquet(paths.item_item, engine="pyarrow")

    # cooc_cnt might not exist depending on your schema
    if "cosine" not in g.columns:
        _maybe_log("[cosine_vs_cooc] cosine missing", verbose)
        return None

    x_col = "cooc_cnt" if "cooc_cnt" in g.columns else None
    if x_col is None:
        _maybe_log("[cosine_vs_cooc] cooc_cnt missing; skipping", verbose)
        return None

    df = g[[x_col, "cosine"]].copy()
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=7)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df[x_col], df["cosine"], s=3, alpha=0.25)
    ax.set_xscale("log")
    ax.set_title("Item-Item Graph: cosine similarity vs co-occurrence (log-x)")
    ax.set_xlabel("Co-occurrence count (log)")
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, alpha=0.25)

    out = paths.out_dir / "C2_graph_cosine_vs_cooc.png"
    _save(fig, out, also_pdf=also_pdf)
    return out


# -------------------------
# D) A/B/n report plot (effect bars + net value)
# -------------------------

def fig_ab_3arm_effects(
    paths: FigurePaths,
    also_pdf: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    if not paths.ab_results_3arm.exists():
        _maybe_log("[ab_effects] missing reports/ab_results_3arm.json", verbose)
        return None

    ab = _read_json(paths.ab_results_3arm)

    tests = ab.get("tests_vs_control", {})
    if not isinstance(tests, dict) or not tests:
        _maybe_log("[ab_effects] tests_vs_control missing/empty", verbose)
        return None

    # Extract effect + lift_pct + p
    arms = []
    effects = []
    lifts = []
    ps = []

    for arm, d in tests.items():
        if not isinstance(d, dict):
            continue
        arms.append(arm)
        effects.append(float(d.get("effect", 0.0)))
        lifts.append(float(d.get("lift_pct", 0.0)))
        ps.append(float(d.get("p_value", 1.0)) if d.get("p_value", None) is not None else 1.0)

    if not arms:
        return None

    # Net value if present
    nv = ab.get("arm_net_value", {}) if isinstance(ab.get("arm_net_value", {}), dict) else {}
    nv_vals = []
    for arm in arms:
        key = f"{arm}_mean_net_value_usd"
        nv_vals.append(float(nv.get(key, 0.0)))

    fig = plt.figure(figsize=(9, 4.8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # Effects (lower is better for revenue_at_risk_usd in your design)
    ax1.bar(arms, effects)
    ax1.axhline(0.0, linestyle="--")
    ax1.set_title("A/B/n: Effect vs Control on Primary Metric (negative = improvement)")
    ax1.set_ylabel("Effect (treatment - control)")

    # annotate p-values lightly
    for i, (arm, p) in enumerate(zip(arms, ps)):
        ax1.text(i, effects[i], f"p={p:.3g}", ha="center", va="bottom")

    # Net value bars
    ax2.bar(arms, nv_vals)
    ax2.axhline(0.0, linestyle="--")
    ax2.set_title("A/B/n: Mean Net Value per User (proxy)")
    ax2.set_ylabel("USD / user")

    out = paths.out_dir / "D_ab_3arm_effects_and_value.png"
    _save(fig, out, also_pdf=also_pdf)
    return out


# -------------------------
# Build all figures + write index.md (markdown)
# -------------------------

def build_all_figures(
    paths: FigurePaths = FigurePaths(),
    also_pdf: bool = False,
    verbose: bool = True,
) -> Path:
    _ensure_dir(paths.out_dir)

    produced: list[Tuple[str, Optional[Path]]] = []

    produced.append(("A) Risk quadrant + top-decile overlay", fig_risk_quadrant(paths, also_pdf=also_pdf, verbose=verbose)))
    produced.append(("B) Ranker metric curves", fig_ranker_metrics_curves(paths, also_pdf=also_pdf, verbose=verbose)))
    produced.append(("C1) Graph degree distribution", fig_graph_degree_distribution(paths, also_pdf=also_pdf, verbose=verbose)))
    produced.append(("C2) Cosine vs co-occurrence", fig_graph_cosine_vs_cooc(paths, also_pdf=also_pdf, verbose=verbose)))
    produced.append(("D) A/B/n effects + net value", fig_ab_3arm_effects(paths, also_pdf=also_pdf, verbose=verbose)))

    idx = paths.out_dir / "index.md"
    lines = []
    lines.append("# NTG Figures (static)\n")
    lines.append("These figures are generated from real project artifacts (parquet/json).\n")
    lines.append("Run: `poetry run python -m ntg.pipelines.make_figures`\n")

    for title, p in produced:
        if p is None:
            lines.append(f"## {title}\n")
            lines.append("_Not generated (missing upstream artifacts)._ \n")
            continue
        rel = p.as_posix()
        lines.append(f"## {title}\n")
        lines.append(f"![]({Path(rel).name})\n")  # relative within same folder

    idx.write_text("\n".join(lines), encoding="utf-8")
    return paths.out_dir
