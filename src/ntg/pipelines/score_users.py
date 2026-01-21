# src/ntg/pipelines/score_users.py
from __future__ import annotations

from ntg.churn import train_model as churn_mod
from ntg.revenue import ltv as ltv_mod
from ntg.revenue import revenue_risk as risk_mod


def main() -> None:
    # Day-6: churn model + scores
    churn_mod.train_and_score(churn_mod.ChurnModelConfig())

    # Day-7: LTV proxy
    ltv_mod.build_user_ltv(ltv_mod.LTVConfig())

    # Day-7: revenue risk
    risk_mod.build_revenue_risk(risk_mod.RevenueRiskConfig())

    print("✅ Day 6–7 complete (Revenue Risk Radar).", flush=True)


if __name__ == "__main__":
    main()
