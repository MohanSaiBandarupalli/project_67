# src/ntg/features/build_user_features.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb


@dataclass(frozen=True)
class UserFeatureConfig:
    # Inputs
    splits_dir: Path = Path("data/processed/splits")
    train_path: Path = Path("data/processed/splits/train.parquet")  # TRAIN only (leakage-safe)

    # Outputs
    out_dir: Path = Path("data/features")
    out_path: Path = Path("data/features/user_features.parquet")
    meta_path: Path = Path("data/features/user_features_meta.json")

    # “recent” windows (days)
    recent_days_7: int = 7
    recent_days_30: int = 30

    # DuckDB tuning
    threads: int = 4
    tmp_dir: Path = Path("data/interim/duckdb_tmp")

    # UX / tests
    enable_progress_bar: bool = False


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_user_features(cfg: Optional[UserFeatureConfig] = None) -> Path:
    """
    Public API expected by unit tests.
    Returns the written parquet path.
    """
    cfg = cfg or UserFeatureConfig()
    run(cfg)
    return cfg.out_path


def run(cfg: UserFeatureConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.train_path.exists():
        raise FileNotFoundError(
            f"Train split not found at {cfg.train_path}. "
            "Run Day-1 dataset/splits build first."
        )

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")
    if cfg.enable_progress_bar:
        con.execute("PRAGMA enable_progress_bar=true;")

    _log("=== Day-2: User features (train-only, leakage-safe) ===")

    # Load train interactions only (leakage-safe)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW train AS
        SELECT
            CAST(user_id AS BIGINT)      AS user_id,
            CAST(item_id AS BIGINT)      AS item_id,
            CAST(rating  AS DOUBLE)      AS rating,
            CAST(timestamp AS TIMESTAMP) AS ts
        FROM read_parquet('{cfg.train_path.as_posix()}');
        """
    )

    # As-of clock from TRAIN only
    con.execute(
        """
        CREATE OR REPLACE VIEW global_clock AS
        SELECT MAX(ts) AS asof_ts FROM train;
        """
    )

    # Feature set:
    # - volume: n_interactions, n_items
    # - preference: rating_mean/std/min/max
    # - activity: active_days, span_days, avg_gap_days
    # - recency: days_since_last
    # - short-window counts: n_last_7d, n_last_30d
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
                        DATE_DIFF(
                            'second',
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
                    SUM(
                        CASE
                            WHEN b.ts >= (g.asof_ts - INTERVAL '{cfg.recent_days_7} days') THEN 1
                            ELSE 0
                        END
                    ) AS n_last_7d,
                    SUM(
                        CASE
                            WHEN b.ts >= (g.asof_ts - INTERVAL '{cfg.recent_days_30} days') THEN 1
                            ELSE 0
                        END
                    ) AS n_last_30d
                FROM base b
                CROSS JOIN global_clock g
                GROUP BY b.user_id
            )
            SELECT
                p.user_id,
                p.n_interactions,
                p.n_items,
                (p.n_items * 1.0) / NULLIF(p.n_interactions, 0) AS item_diversity,

                CAST(p.rating_mean AS DOUBLE) AS rating_mean,
                COALESCE(CAST(p.rating_std AS DOUBLE), 0.0) AS rating_std,
                CAST(p.rating_min AS DOUBLE) AS rating_min,
                CAST(p.rating_max AS DOUBLE) AS rating_max,

                CAST(p.active_days AS BIGINT) AS active_days,
                CAST(r.span_days AS DOUBLE) AS span_days,
                CAST(r.days_since_last AS DOUBLE) AS days_since_last,

                CAST(COALESCE(g.avg_gap_days, r.span_days) AS DOUBLE) AS avg_gap_days,
                CAST(COALESCE(g.median_gap_days, r.span_days) AS DOUBLE) AS median_gap_days,

                CAST(rc.n_last_7d  AS BIGINT) AS n_last_7d,
                CAST(rc.n_last_30d AS BIGINT) AS n_last_30d
            FROM per_user p
            LEFT JOIN gaps g ON g.user_id = p.user_id
            LEFT JOIN recency r ON r.user_id = p.user_id
            LEFT JOIN recent_counts rc ON rc.user_id = p.user_id
        ) TO '{cfg.out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # metadata (more production-like)
    stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT user_id) AS n_users,
            MIN(ts) AS min_ts,
            MAX(ts) AS max_ts
        FROM train;
        """
    ).fetchone()

    columns = [
        "user_id",
        "n_interactions",
        "n_items",
        "item_diversity",
        "rating_mean",
        "rating_std",
        "rating_min",
        "rating_max",
        "active_days",
        "span_days",
        "days_since_last",
        "avg_gap_days",
        "median_gap_days",
        "n_last_7d",
        "n_last_30d",
    ]

    meta: Dict[str, Any] = {
        "strategy": "train_only_user_features",
        "source_train_path": str(cfg.train_path),
        "out_path": str(cfg.out_path),
        "n_rows_train": int(stats[0]),
        "n_users": int(stats[1]),
        "train_min_ts": str(stats[2]) if stats[2] is not None else None,
        "train_max_ts": str(stats[3]) if stats[3] is not None else None,
        "asof_clock": "max(train.ts)",
        "recent_days_7": cfg.recent_days_7,
        "recent_days_30": cfg.recent_days_30,
        "leakage_safe": True,
        "columns": columns,
        "notes": "All features computed from TRAIN only. Use these for churn/LTV/ranking without timestamp leakage.",
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(f"✅ Wrote: {cfg.out_path}")
    _log(f"✅ Meta : {cfg.meta_path}")


def main() -> None:
    build_user_features(UserFeatureConfig())


if __name__ == "__main__":
    main()
