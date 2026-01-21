from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class UserFeatureConfig:
    splits_dir: Path = Path("data/processed/splits")
    # IMPORTANT: Use TRAIN only to avoid leakage in features.
    train_path: Path = Path("data/processed/splits/train.parquet")

    out_dir: Path = Path("data/features")
    out_path: Path = Path("data/features/user_features.parquet")
    meta_path: Path = Path("data/features/user_features_meta.json")

    # time windows for “recent” behavior
    recent_days_7: int = 7
    recent_days_30: int = 30

    # DuckDB tuning
    threads: int = 4
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def run(cfg: UserFeatureConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.train_path.exists():
        raise FileNotFoundError(
            f"Train split not found at {cfg.train_path}. Run Day-1 build_dataset_duckdb first."
        )

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")
    con.execute("PRAGMA enable_progress_bar=true;")

    # Load train interactions only (leakage-safe)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW train AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(item_id AS BIGINT) AS item_id,
            CAST(rating  AS DOUBLE) AS rating,
            CAST(timestamp AS TIMESTAMP) AS ts
        FROM read_parquet('{cfg.train_path.as_posix()}');
        """
    )

    # Global “as-of” time based on train
    con.execute(
        """
        CREATE OR REPLACE VIEW global_clock AS
        SELECT MAX(ts) AS asof_ts FROM train;
        """
    )

    # Feature set (robust & interview-friendly):
    # - volume: n_interactions, n_items
    # - preference: rating_mean/std/min/max
    # - activity: active_days, span_days, avg_gap_days
    # - recency: days_since_last
    # - short-window counts: last_7d, last_30d
    con.execute(
        f"""
        COPY (
            WITH base AS (
                SELECT
                    t.*,
                    CAST(t.ts AS DATE) AS d
                FROM train t
            ),
            per_user AS (
                SELECT
                    user_id,
                    COUNT(*) AS n_interactions,
                    COUNT(DISTINCT item_id) AS n_items,
                    AVG(rating) AS rating_mean,
                    STDDEV_SAMP(rating) AS rating_std,
                    MIN(rating) AS rating_min,
                    MAX(rating) AS rating_max,
                    MIN(ts) AS first_ts,
                    MAX(ts) AS last_ts,
                    COUNT(DISTINCT d) AS active_days
                FROM base
                GROUP BY user_id
            ),
            gaps AS (
                SELECT
                    user_id,
                    AVG(gap_days) AS avg_gap_days,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_days) AS median_gap_days
                FROM (
                    SELECT
                        user_id,
                        DATE_DIFF('second',
                            LAG(ts) OVER (PARTITION BY user_id ORDER BY ts, item_id),
                            ts
                        ) / 86400.0 AS gap_days
                    FROM train
                )
                WHERE gap_days IS NOT NULL
                GROUP BY user_id
            ),
            recency AS (
                SELECT
                    p.user_id,
                    DATE_DIFF('second', p.last_ts, g.asof_ts) / 86400.0 AS days_since_last,
                    DATE_DIFF('second', p.first_ts, p.last_ts) / 86400.0 AS span_days
                FROM per_user p
                CROSS JOIN global_clock g
            ),
            recent_counts AS (
                SELECT
                    b.user_id,
                    SUM(CASE WHEN b.ts >= (g.asof_ts - INTERVAL '{cfg.recent_days_7} days') THEN 1 ELSE 0 END) AS n_last_7d,
                    SUM(CASE WHEN b.ts >= (g.asof_ts - INTERVAL '{cfg.recent_days_30} days') THEN 1 ELSE 0 END) AS n_last_30d
                FROM base b
                CROSS JOIN global_clock g
                GROUP BY b.user_id
            )
            SELECT
                p.user_id,
                p.n_interactions,
                p.n_items,
                (p.n_items * 1.0) / NULLIF(p.n_interactions, 0) AS item_diversity,
                p.rating_mean,
                COALESCE(p.rating_std, 0.0) AS rating_std,
                p.rating_min,
                p.rating_max,
                p.active_days,
                r.span_days,
                r.days_since_last,
                COALESCE(g.avg_gap_days, r.span_days) AS avg_gap_days,
                COALESCE(g.median_gap_days, r.span_days) AS median_gap_days,
                rc.n_last_7d,
                rc.n_last_30d
            FROM per_user p
            LEFT JOIN gaps g ON g.user_id = p.user_id
            LEFT JOIN recency r ON r.user_id = p.user_id
            LEFT JOIN recent_counts rc ON rc.user_id = p.user_id
        ) TO '{cfg.out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # metadata
    n_users = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT user_id FROM train);").fetchone()[0]
    meta = {
        "source": str(cfg.train_path),
        "n_users": int(n_users),
        "recent_days_7": cfg.recent_days_7,
        "recent_days_30": cfg.recent_days_30,
        "leakage_safe": True,
        "note": "Features computed from TRAIN split only (as-of clock = max(train.ts)).",
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ Wrote: {cfg.out_path}")
    print(f"✅ Meta : {cfg.meta_path}")


if __name__ == "__main__":
    run(UserFeatureConfig())
