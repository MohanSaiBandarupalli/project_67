from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class ItemFeatureConfig:
    train_path: Path = Path("data/processed/splits/train.parquet")
    out_dir: Path = Path("data/features")
    out_path: Path = Path("data/features/item_features.parquet")
    meta_path: Path = Path("data/features/item_features_meta.json")

    recent_days_7: int = 7
    recent_days_30: int = 30

    threads: int = 4
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def run(cfg: ItemFeatureConfig) -> None:
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

    con.execute("CREATE OR REPLACE VIEW global_clock AS SELECT MAX(ts) AS asof_ts FROM train;")

    # Item features:
    # - popularity: n_interactions, n_users
    # - rating: mean/std
    # - temporal: first_seen, last_seen, days_since_last
    # - recent_velocity: counts last 7/30 days
    con.execute(
        f"""
        COPY (
            WITH per_item AS (
                SELECT
                    item_id,
                    COUNT(*) AS n_interactions,
                    COUNT(DISTINCT user_id) AS n_users,
                    AVG(rating) AS rating_mean,
                    STDDEV_SAMP(rating) AS rating_std,
                    MIN(ts) AS first_ts,
                    MAX(ts) AS last_ts
                FROM train
                GROUP BY item_id
            ),
            recency AS (
                SELECT
                    p.item_id,
                    DATE_DIFF('second', p.last_ts, g.asof_ts) / 86400.0 AS days_since_last
                FROM per_item p
                CROSS JOIN global_clock g
            ),
            recent_counts AS (
                SELECT
                    t.item_id,
                    SUM(CASE WHEN t.ts >= (g.asof_ts - INTERVAL '{cfg.recent_days_7} days') THEN 1 ELSE 0 END) AS n_last_7d,
                    SUM(CASE WHEN t.ts >= (g.asof_ts - INTERVAL '{cfg.recent_days_30} days') THEN 1 ELSE 0 END) AS n_last_30d
                FROM train t
                CROSS JOIN global_clock g
                GROUP BY t.item_id
            )
            SELECT
                p.item_id,
                p.n_interactions,
                p.n_users,
                p.rating_mean,
                COALESCE(p.rating_std, 0.0) AS rating_std,
                p.first_ts,
                p.last_ts,
                r.days_since_last,
                rc.n_last_7d,
                rc.n_last_30d,
                (rc.n_last_30d * 1.0) / NULLIF(p.n_interactions, 0) AS recent_share_30d
            FROM per_item p
            LEFT JOIN recency r ON r.item_id = p.item_id
            LEFT JOIN recent_counts rc ON rc.item_id = p.item_id
        ) TO '{cfg.out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    n_items = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT item_id FROM train);").fetchone()[0]
    meta = {
        "source": str(cfg.train_path),
        "n_items": int(n_items),
        "recent_days_7": cfg.recent_days_7,
        "recent_days_30": cfg.recent_days_30,
        "leakage_safe": True,
        "note": "Features computed from TRAIN split only (as-of clock = max(train.ts)).",
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ Wrote: {cfg.out_path}")
    print(f"✅ Meta : {cfg.meta_path}")


if __name__ == "__main__":
    run(ItemFeatureConfig())
