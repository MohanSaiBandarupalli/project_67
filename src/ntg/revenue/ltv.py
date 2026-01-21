# src/ntg/revenue/ltv.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class LTVConfig:
    train_split_path: Path = Path("data/processed/splits/train.parquet")

    out_dir: Path = Path("outputs/risk")
    out_ltv_path: Path = Path("outputs/risk/user_ltv.parquet")
    out_meta_path: Path = Path("outputs/risk/user_ltv_meta.json")

    # LTV proxy assumptions (prototype, defensible)
    monthly_price_usd: float = 15.49
    max_months: int = 24
    min_months: int = 1

    # DuckDB tuning
    threads: int = 8
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_user_ltv(cfg: LTVConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.train_split_path.exists():
        raise FileNotFoundError(f"Missing {cfg.train_split_path}. Run Day-1 first.")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")

    _log("=== Day-7: LTV proxy ===")
    _log("[1/2] Computing user engagement stats from TRAIN")

    # Read raw first (timestamp may be TIMESTAMP, BIGINT epoch, DOUBLE, etc.)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW raw_tr AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(item_id AS BIGINT) AS item_id,
            timestamp AS ts_raw
        FROM read_parquet('{cfg.train_split_path.as_posix()}');
        """
    )

    # Normalize ts_raw -> epoch seconds (DOUBLE) with strict typing
    con.execute(
        """
        CREATE OR REPLACE VIEW tr AS
        SELECT
            user_id,
            item_id,
            CAST(
                CASE
                    WHEN typeof(ts_raw) LIKE 'TIMESTAMP%' THEN epoch(ts_raw)
                    WHEN typeof(ts_raw) = 'DATE' THEN epoch(CAST(ts_raw AS TIMESTAMP))
                    WHEN typeof(ts_raw) IN ('BIGINT','INTEGER','SMALLINT','TINYINT')
                        THEN CAST(ts_raw AS DOUBLE)
                    WHEN typeof(ts_raw) IN ('DOUBLE','FLOAT','REAL')
                        THEN CAST(ts_raw AS DOUBLE)
                    ELSE epoch(CAST(ts_raw AS TIMESTAMP))
                END
            AS DOUBLE) AS ts
        FROM raw_tr;
        """
    )

    # Compute LTV proxy using TRAIN-only engagement
    con.execute(
        f"""
        COPY (
            WITH mx AS (SELECT MAX(ts) AS max_ts FROM tr),
            agg AS (
                SELECT
                    user_id,
                    COUNT(*) AS n_interactions,
                    COUNT(DISTINCT item_id) AS n_distinct_items,
                    (SELECT max_ts FROM mx) - MAX(ts) AS recency_seconds
                FROM tr
                GROUP BY user_id
            )
            SELECT
                user_id,
                n_interactions,
                n_distinct_items,
                (recency_seconds / 86400.0) AS recency_days,

                LEAST(
                    {cfg.max_months},
                    GREATEST(
                        {cfg.min_months},
                        CAST(ROUND(
                            2
                            + 1.2 * LN(1 + n_interactions)
                            + 0.8 * LN(1 + n_distinct_items)
                            - 0.08 * (recency_seconds / 86400.0)
                        ) AS BIGINT)
                    )
                ) AS expected_months,

                ({cfg.monthly_price_usd} * LEAST(
                    {cfg.max_months},
                    GREATEST(
                        {cfg.min_months},
                        CAST(ROUND(
                            2
                            + 1.2 * LN(1 + n_interactions)
                            + 0.8 * LN(1 + n_distinct_items)
                            - 0.08 * (recency_seconds / 86400.0)
                        ) AS BIGINT)
                    )
                )) AS ltv_usd
            FROM agg
        ) TO '{cfg.out_ltv_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    n_users = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{cfg.out_ltv_path.as_posix()}')"
    ).fetchone()[0]

    meta = {
        "train_split": str(cfg.train_split_path),
        "monthly_price_usd": cfg.monthly_price_usd,
        "min_months": cfg.min_months,
        "max_months": cfg.max_months,
        "n_users": int(n_users),
        "notes": "Prototype LTV proxy computed from TRAIN engagement only (leakage-safe). Timestamp normalized to epoch seconds.",
    }
    cfg.out_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(f"✅ Wrote: {cfg.out_ltv_path}")
    _log(f"✅ Meta : {cfg.out_meta_path}")


if __name__ == "__main__":
    build_user_ltv(LTVConfig())
