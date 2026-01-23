# src/ntg/revenue/revenue_risk.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


def _parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    """
    Return column names for a parquet file using DuckDB.
    """
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')"
    ).fetchall()
    # DESCRIBE returns rows like: (column_name, type, null, key, default, extra)
    return [r[0] for r in rows]


def _pick_first(existing: Iterable[str], candidates: Iterable[str]) -> str | None:
    existing_set = set(existing)
    for c in candidates:
        if c in existing_set:
            return c
    return None


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

    # --- Detect churn prob column (supports multiple schemas) ---
    churn_cols = _parquet_columns(con, cfg.churn_scores_path)
    # Prefer your project conventions first; include current observed "churn_prob"
    churn_prob_col = _pick_first(
        churn_cols,
        candidates=["p_churn", "churn_prob", "prob_churn", "p", "probability", "score"],
    )
    if churn_prob_col is None:
        raise ValueError(
            f"Could not find churn probability column in {cfg.churn_scores_path}. "
            f"Found columns: {churn_cols}. "
            f"Expected one of: p_churn, churn_prob, prob_churn, probability, score."
        )

    # user_id should exist; still keep it strict for sanity
    if "user_id" not in set(churn_cols):
        raise ValueError(
            f"Missing required column 'user_id' in {cfg.churn_scores_path}. "
            f"Found columns: {churn_cols}"
        )

    # NOTE: Qualify columns with a table alias to avoid DuckDB binder ambiguity.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW churn AS
        SELECT
            CAST(t.user_id AS BIGINT) AS user_id,
            CAST(t.{churn_prob_col} AS DOUBLE) AS p_churn
        FROM read_parquet('{cfg.churn_scores_path.as_posix()}') AS t;
        """
    )

    # --- LTV view (qualify with alias too) ---
    con.execute(
        f"""
        CREATE OR REPLACE VIEW ltv AS
        SELECT
            CAST(t.user_id AS BIGINT) AS user_id,
            CAST(t.ltv_usd AS DOUBLE) AS ltv_usd,
            CAST(t.expected_months AS BIGINT) AS expected_months,
            CAST(t.n_interactions AS BIGINT) AS n_interactions,
            CAST(t.n_distinct_items AS BIGINT) AS n_distinct_items,
            CAST(t.recency_days AS DOUBLE) AS recency_days
        FROM read_parquet('{cfg.ltv_path.as_posix()}') AS t;
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
        "schema": {
            "churn_prob_source_column": churn_prob_col,
            "churn_scores_path": str(cfg.churn_scores_path),
            "ltv_path": str(cfg.ltv_path),
        },
    }
    cfg.out_summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    _log(f"✅ Wrote: {cfg.out_path}")
    _log(f"✅ Wrote: {cfg.out_top_path}")
    _log(f"✅ Summary: {cfg.out_summary_path}")


if __name__ == "__main__":
    build_revenue_risk(RevenueRiskConfig())
