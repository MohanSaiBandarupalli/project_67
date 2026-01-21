from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ntg.experiments.assign import AssignmentConfig, assign_variants
from ntg.experiments.design import ExperimentDesign, required_sample_size_continuous, ztest_diff_means


@dataclass(frozen=True)
class ABSimConfig:
    # Inputs
    user_risk_path: Path = Path("outputs/risk/user_risk.parquet")
    interventions_path: Path = Path("outputs/interventions/interventions.parquet")

    # Outputs
    out_json: Path = Path("reports/ab_results.json")
    out_md: Path = Path("reports/ab_experiment_design.md")

    # Experiment knobs
    experiment_id: str = "ntg_ab_v1"
    treatment_share: float = 0.50

    # Primary metric: revenue_at_risk_usd (lower is better)
    # We simulate treatment using p_churn_new if available (from interventions.parquet)
    design: ExperimentDesign = ExperimentDesign(alpha=0.05, power=0.80, mde_pct=0.03)

    random_state: int = 42


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_design_md(baseline_mean: float, baseline_std: float, n_per_group: int, cfg: ABSimConfig) -> None:
    md = f"""# A/B Experiment Design (Day 10–11)

## Objective
Validate that **NTG-driven interventions** reduce churn risk and **reduce revenue-at-risk**.

## Variants
- **Control**: current experience (no intervention)
- **Treatment**: intervention policy (discount/content boost/email nudge)

## Primary Metric
- **Revenue-at-risk (USD)** per user (lower is better)  
  Computed as: `ltv_usd * p_churn`

## Secondary Metrics
- Mean churn probability `p_churn`
- Net value proxy (if using intervention costs): `risk_reduction - cost`

## Randomization
Deterministic hashing by `user_id`:
- experiment_id = `{cfg.experiment_id}`
- split = {int(cfg.treatment_share*100)}/{int((1-cfg.treatment_share)*100)}

## Sample Size (normal approx)
Baseline mean: **{baseline_mean:,.4f}**  
Baseline std: **{baseline_std:,.4f}**  
MDE: **{cfg.design.mde_pct*100:.2f}%**  
alpha: **{cfg.design.alpha}**, power: **{cfg.design.power}**  

Required sample size (per group): **{n_per_group:,}**

## Notes
- This is an **offline simulation** using trained risk scores.
- Online A/B would require instrumentation + guardrails + rollout plan.
"""
    cfg.out_md.write_text(md, encoding="utf-8")


def run_ab_simulation(cfg: ABSimConfig) -> Dict[str, float]:
    np.random.seed(cfg.random_state)

    if not cfg.user_risk_path.exists():
        raise FileNotFoundError(f"Missing {cfg.user_risk_path}. Run Day 6–7 first.")
    if not cfg.interventions_path.exists():
        raise FileNotFoundError(f"Missing {cfg.interventions_path}. Run Day 8–9 first.")

    _log("=== Day 10–11: A/B Simulator ===")
    _log("[1/5] Loading user_risk + interventions outputs")

    base = pd.read_parquet(cfg.user_risk_path)
    inter = pd.read_parquet(cfg.interventions_path)

    # Join on user_id
    df = base.merge(
        inter[["user_id", "p_churn_new", "revenue_at_risk_new_usd", "intervention", "intervention_cost_usd"]],
        on="user_id",
        how="left",
    )

    # If intervention columns missing, fallback to baseline
    if "p_churn_new" not in df.columns:
        df["p_churn_new"] = df["p_churn"]
    df["p_churn_new"] = df["p_churn_new"].fillna(df["p_churn"])

    if "revenue_at_risk_new_usd" not in df.columns:
        df["revenue_at_risk_new_usd"] = df["revenue_at_risk_usd"]
    df["revenue_at_risk_new_usd"] = df["revenue_at_risk_new_usd"].fillna(df["revenue_at_risk_usd"])

    if "intervention_cost_usd" not in df.columns:
        df["intervention_cost_usd"] = 0.0
    df["intervention_cost_usd"] = df["intervention_cost_usd"].fillna(0.0)

    # Assign users to variants deterministically
    _log("[2/5] Assigning control/treatment (deterministic bucketing)")
    df = assign_variants(df, AssignmentConfig(experiment_id=cfg.experiment_id, treatment_share=cfg.treatment_share))

    # Observed metric for each variant
    # Control uses baseline risk; Treatment uses post-intervention risk
    _log("[3/5] Building observed metrics per variant")
    control_metric = df.loc[df["variant"] == "control", "revenue_at_risk_usd"].to_numpy(dtype=float)
    treat_metric = df.loc[df["variant"] == "treatment", "revenue_at_risk_new_usd"].to_numpy(dtype=float)

    # Design: baseline distribution for sample size planning
    baseline_mean = float(df["revenue_at_risk_usd"].mean())
    baseline_std = float(df["revenue_at_risk_usd"].std(ddof=1))
    n_per_group = required_sample_size_continuous(baseline_mean, baseline_std, cfg.design)

    _write_design_md(baseline_mean, baseline_std, n_per_group, cfg)

    # Stats test (lower is better, but we report signed effect)
    _log("[4/5] Running z-test on mean revenue-at-risk")
    res = ztest_diff_means(control_metric, treat_metric)

    # Also report churn prob movement
    control_p = df.loc[df["variant"] == "control", "p_churn"].to_numpy(dtype=float)
    treat_p = df.loc[df["variant"] == "treatment", "p_churn_new"].to_numpy(dtype=float)
    churn_res = ztest_diff_means(control_p, treat_p)

    # Net value proxy (risk reduction - cost) for treatment arm
    df["net_value_usd"] = (df["revenue_at_risk_usd"] - df["revenue_at_risk_new_usd"]) - df["intervention_cost_usd"]
    net_treat = df.loc[df["variant"] == "treatment", "net_value_usd"].to_numpy(dtype=float)

    out = {
        "experiment_id": cfg.experiment_id,
        "n_total": int(len(df)),
        "n_control": int((df["variant"] == "control").sum()),
        "n_treatment": int((df["variant"] == "treatment").sum()),
        "primary_metric": "revenue_at_risk_usd (lower is better)",
        "design": {
            "alpha": cfg.design.alpha,
            "power": cfg.design.power,
            "mde_pct": cfg.design.mde_pct,
            "required_n_per_group": int(n_per_group),
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        },
        "primary_test": res,
        "secondary_test_churn_prob": churn_res,
        "treatment_net_value": {
            "mean_net_value_usd": float(np.mean(net_treat)) if len(net_treat) else float("nan"),
            "total_net_value_usd": float(np.sum(net_treat)) if len(net_treat) else float("nan"),
        },
    }

    _log("[5/5] Writing report JSON")
    cfg.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _log(f"✅ Wrote: {cfg.out_json}")
    _log(f"✅ Wrote: {cfg.out_md}")

    return out


if __name__ == "__main__":
    run_ab_simulation(ABSimConfig())
