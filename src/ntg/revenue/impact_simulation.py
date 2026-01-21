from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InterventionConfig:
    # inputs
    user_risk_path: Path = Path("outputs/risk/user_risk.parquet")
    user_risk_top_path: Path = Path("outputs/risk/user_risk_top.parquet")

    # outputs
    out_dir: Path = Path("outputs/interventions")
    out_parquet: Path = Path("outputs/interventions/interventions.parquet")
    out_json: Path = Path("outputs/interventions/intervention_summary.json")
    out_md: Path = Path("reports/intervention_report.md")

    # budget + policies (prototype knobs)
    monthly_price_usd: float = 15.49

    # segment thresholds
    high_risk_p: float = 0.70
    mid_risk_p: float = 0.40

    # intervention effects (expected lift)
    # (these are tunable assumptions; keep them explicit & logged)
    discount_pct: float = 0.20           # 20% off for 2 months
    discount_months: int = 2
    discount_churn_mult: float = 0.70    # reduces churn prob by 30% (p *= 0.70)

    content_boost_churn_mult: float = 0.85  # better recs reduce churn 15%
    email_nudge_churn_mult: float = 0.92     # email nudge reduces churn 8%

    # targeting caps (so we don't "discount everyone")
    max_discount_users: int = 25000
    max_content_boost_users: int = 50000
    max_email_users: int = 100000

    random_state: int = 42


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _pick_segments(df: pd.DataFrame, cfg: InterventionConfig) -> pd.DataFrame:
    """
    Assign one primary intervention per user using simple, explainable rules.
    Priority:
      1) High churn + high LTV => discount (save whales)
      2) High churn + mid LTV  => content boost (fix relevance)
      3) Mid churn             => email nudge
      4) Low churn             => no action
    """
    df = df.copy()

    # Basic buckets
    df["risk_tier"] = np.select(
        [df["p_churn"] >= cfg.high_risk_p, df["p_churn"] >= cfg.mid_risk_p],
        ["high", "mid"],
        default="low",
    )

    # LTV tiers by quantile (robust across datasets)
    q80 = df["ltv_usd"].quantile(0.80)
    q50 = df["ltv_usd"].quantile(0.50)

    df["ltv_tier"] = np.select(
        [df["ltv_usd"] >= q80, df["ltv_usd"] >= q50],
        ["high", "mid"],
        default="low",
    )

    # Default
    df["intervention"] = "none"

    # Assign by rule
    cond_discount = (df["risk_tier"] == "high") & (df["ltv_tier"] == "high")
    cond_content = (df["risk_tier"] == "high") & (df["ltv_tier"] != "high")
    cond_email = (df["risk_tier"] == "mid")

    df.loc[cond_discount, "intervention"] = "discount_2mo"
    df.loc[cond_content, "intervention"] = "content_boost"
    df.loc[cond_email, "intervention"] = "email_nudge"

    # Apply caps (deterministic)
    df = df.sort_values(["revenue_at_risk_usd", "p_churn"], ascending=[False, False]).reset_index(drop=True)

    def cap(label: str, max_n: int) -> None:
        idx = df.index[df["intervention"] == label].to_numpy()
        if len(idx) > max_n:
            drop_idx = idx[max_n:]
            df.loc[drop_idx, "intervention"] = "none"

    cap("discount_2mo", cfg.max_discount_users)
    cap("content_boost", cfg.max_content_boost_users)
    cap("email_nudge", cfg.max_email_users)

    return df


def _simulate(df: pd.DataFrame, cfg: InterventionConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute new churn probability and expected revenue-at-risk after interventions.
    """
    df = df.copy()

    base_p = df["p_churn"].to_numpy(dtype=float)

    mult = np.ones_like(base_p)

    mult[df["intervention"] == "discount_2mo"] *= cfg.discount_churn_mult
    mult[df["intervention"] == "content_boost"] *= cfg.content_boost_churn_mult
    mult[df["intervention"] == "email_nudge"] *= cfg.email_nudge_churn_mult

    new_p = _clamp01(base_p * mult)

    df["p_churn_new"] = new_p.astype("float32")
    df["revenue_at_risk_new_usd"] = (df["ltv_usd"].to_numpy(dtype=float) * new_p).astype("float32")

    # Cost model (simple, explicit)
    # discount cost = discount_pct * monthly_price * months for treated users
    df["intervention_cost_usd"] = 0.0
    mask_disc = df["intervention"] == "discount_2mo"
    df.loc[mask_disc, "intervention_cost_usd"] = (
        cfg.discount_pct * cfg.monthly_price_usd * cfg.discount_months
    )

    # Expected recovered revenue-at-risk = base_risk - new_risk - cost
    df["risk_reduction_usd"] = (df["revenue_at_risk_usd"] - df["revenue_at_risk_new_usd"]).astype("float32")
    df["net_value_usd"] = (df["risk_reduction_usd"] - df["intervention_cost_usd"]).astype("float32")

    summary = {
        "n_users": float(len(df)),
        "base_total_risk_usd": float(df["revenue_at_risk_usd"].sum()),
        "new_total_risk_usd": float(df["revenue_at_risk_new_usd"].sum()),
        "risk_reduced_usd": float(df["risk_reduction_usd"].sum()),
        "total_cost_usd": float(df["intervention_cost_usd"].sum()),
        "net_value_usd": float(df["net_value_usd"].sum()),
        "treated_discount": float((df["intervention"] == "discount_2mo").sum()),
        "treated_content_boost": float((df["intervention"] == "content_boost").sum()),
        "treated_email": float((df["intervention"] == "email_nudge").sum()),
        "treated_any": float((df["intervention"] != "none").sum()),
        "avg_p_churn": float(df["p_churn"].mean()),
        "avg_p_churn_new": float(df["p_churn_new"].mean()),
    }

    return df, summary


def _write_report_md(summary: Dict[str, float], out_md: Path) -> None:
    md = f"""# Intervention Report (Day 8–9)

## Executive Summary
This simulation applies **rule-based retention interventions** to users flagged by the Revenue Risk Radar and estimates impact on churn probability and revenue-at-risk.

### Key results
- Users scored: **{int(summary["n_users"]):,}**
- Treated users: **{int(summary["treated_any"]):,}**
  - Discount (2 months): **{int(summary["treated_discount"]):,}**
  - Content boost: **{int(summary["treated_content_boost"]):,}**
  - Email nudge: **{int(summary["treated_email"]):,}**

### Financial impact (expected)
- Base revenue-at-risk: **${summary["base_total_risk_usd"]:,.2f}**
- Post-intervention revenue-at-risk: **${summary["new_total_risk_usd"]:,.2f}**
- Risk reduced: **${summary["risk_reduced_usd"]:,.2f}**
- Intervention cost: **${summary["total_cost_usd"]:,.2f}**
- **Net value (risk reduced − cost): ${summary["net_value_usd"]:,.2f}**

### Churn probability shift
- Avg p(churn): **{summary["avg_p_churn"]:.4f}**
- Avg p(churn) after: **{summary["avg_p_churn_new"]:.4f}**

## Notes
- This is a **prototype**, so effect sizes are explicit assumptions (tunable).
- All calculations are **leakage-safe**: derived from TRAIN-only features and models.
"""
    out_md.write_text(md, encoding="utf-8")


def run_intervention_simulation(cfg: InterventionConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    if not cfg.user_risk_path.exists():
        raise FileNotFoundError(f"Missing {cfg.user_risk_path}. Run Day 6–7 first.")

    _log("=== Day 8–9: Intervention Simulator ===")
    _log("[1/4] Loading user risk table")
    df = pd.read_parquet(cfg.user_risk_path)

    needed = {"user_id", "p_churn", "ltv_usd", "revenue_at_risk_usd"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"user_risk.parquet missing columns: {sorted(missing)}")

    _log("[2/4] Assigning interventions (rule-based)")
    df = _pick_segments(df, cfg)

    _log("[3/4] Simulating churn reduction + net value")
    df2, summary = _simulate(df, cfg)

    _log("[4/4] Writing outputs")
    df2.to_parquet(cfg.out_parquet, index=False)

    cfg.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report_md(summary, cfg.out_md)

    _log(f"✅ Wrote: {cfg.out_parquet}")
    _log(f"✅ Summary: {cfg.out_json}")
    _log(f"✅ Report: {cfg.out_md}")


if __name__ == "__main__":
    run_intervention_simulation(InterventionConfig())
