# src/ntg/revenue/revenue_risk.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class RevenueRiskConfig:
    churn_scores_path: Path = Path("outputs/risk/churn_scores.parquet")
    ltv_path: Path = Path("outputs/risk/user_ltv.parquet")

    out_dir: Path = Path("outputs/risk")
    out_path: Path = Path("outputs/risk/user_risk.parquet")
    out_top_path: Path = Path("outputs/risk/user_risk_top.parquet")
    out_summary_path: Path = Path("reports/revenue_risk_summary.json")

    top_frac: float = 0.02  # top 2% users by revenue_at_risk

    threads: int = 8
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_revenue_risk(cfg: RevenueRiskConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.churn_scores_path.exists():
        raise FileNotFoundError(f"Missing {cfg.churn_scores_path}. Run churn model first.")
    if not cfg.ltv_path.exists():
        raise FileNotFoundError(f"Missing {cfg.ltv_path}. Run LTV step first.")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")

    _log("=== Day-7: Revenue Risk Radar ===")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW churn AS
        SELECT CAST(user_id AS BIGINT) AS user_id, CAST(p_churn AS DOUBLE) AS p_churn
        FROM read_parquet('{cfg.churn_scores_path.as_posix()}');
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW ltv AS
        SELECT CAST(user_id AS BIGINT) AS user_id,
               CAST(ltv_usd AS DOUBLE) AS ltv_usd,
               CAST(expected_months AS BIGINT) AS expected_months,
               CAST(n_interactions AS BIGINT) AS n_interactions,
               CAST(n_distinct_items AS BIGINT) AS n_distinct_items,
               CAST(recency_days AS DOUBLE) AS recency_days
        FROM read_parquet('{cfg.ltv_path.as_posix()}');
        """
    )

    # full table
    con.execute(
        f"""
        COPY (
            WITH base AS (
                SELECT
                    c.user_id,
                    c.p_churn,
                    l.ltv_usd,
                    (c.p_churn * l.ltv_usd) AS revenue_at_risk_usd,
                    l.expected_months,
                    l.n_interactions,
                    l.n_distinct_items,
                    l.recency_days
                FROM churn c
                JOIN ltv l USING(user_id)
            )
            SELECT
                *,
                CASE
                    WHEN revenue_at_risk_usd >= 100 THEN 'critical'
                    WHEN revenue_at_risk_usd >= 50 THEN 'high'
                    WHEN revenue_at_risk_usd >= 20 THEN 'medium'
                    ELSE 'low'
                END AS risk_bucket
            FROM base
        ) TO '{cfg.out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # top users
    con.execute(
        f"""
        COPY (
            WITH r AS (
                SELECT * FROM read_parquet('{cfg.out_path.as_posix()}')
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (ORDER BY revenue_at_risk_usd DESC, p_churn DESC, ltv_usd DESC) AS rn,
                    COUNT(*) OVER () AS n
                FROM r
            )
            SELECT * FROM ranked
            WHERE rn <= CAST(CEIL(n * {cfg.top_frac}) AS BIGINT)
        ) TO '{cfg.out_top_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # summary
    summary = con.execute(
        f"""
        WITH r AS (SELECT * FROM read_parquet('{cfg.out_path.as_posix()}'))
        SELECT
            COUNT(*) AS n_users,
            AVG(p_churn) AS avg_p_churn,
            AVG(ltv_usd) AS avg_ltv,
            SUM(revenue_at_risk_usd) AS total_revenue_at_risk_usd,
            SUM(CASE WHEN risk_bucket IN ('high','critical') THEN revenue_at_risk_usd ELSE 0 END) AS high_critical_risk_usd
        FROM r;
        """
    ).fetchone()

    out = {
        "n_users": int(summary[0]),
        "avg_p_churn": float(summary[1]),
        "avg_ltv_usd": float(summary[2]),
        "total_revenue_at_risk_usd": float(summary[3]),
        "high_critical_risk_usd": float(summary[4]),
        "top_frac": cfg.top_frac,
        "notes": "Revenue Risk = P(churn) * LTV proxy. Prototype uses MovieLens behavior; in production replace LTV with billing.",
    }
    cfg.out_summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    _log(f"✅ Wrote: {cfg.out_path}")
    _log(f"✅ Wrote: {cfg.out_top_path}")
    _log(f"✅ Summary: {cfg.out_summary_path}")


if __name__ == "__main__":
    build_revenue_risk(RevenueRiskConfig())
