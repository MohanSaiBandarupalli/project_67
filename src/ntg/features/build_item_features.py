# src/ntg/features/build_item_features.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb


@dataclass(frozen=True)
class ItemFeatureConfig:
    """
    Leakage-safe item feature builder.

    Uses TRAIN split only and defines an "as-of" clock as max(train.ts),
    then computes recency/velocity features relative to that clock.

    Outputs:
      - data/features/item_features.parquet
      - data/features/item_features_meta.json
    """

    # Inputs
    train_path: Path = Path("data/processed/splits/train.parquet")

    # Outputs
    out_dir: Path = Path("data/features")
    out_path: Path = Path("data/features/item_features.parquet")
    meta_path: Path = Path("data/features/item_features_meta.json")

    # Feature windows
    recent_days_7: int = 7
    recent_days_30: int = 30

    # DuckDB tuning
    threads: int = 4
    tmp_dir: Path = Path("data/interim/duckdb_tmp")
    enable_progress_bar: bool = False


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_item_features(cfg: Optional[ItemFeatureConfig] = None) -> Path:
    """
    Public API expected by unit tests.
    Returns path to the written parquet.
    """
    cfg = cfg or ItemFeatureConfig()
    run(cfg)
    return cfg.out_path


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
    if cfg.enable_progress_bar:
        con.execute("PRAGMA enable_progress_bar=true;")

    _log("=== Day-2: Item features (train-only) ===")
    _log(f"[duckdb] reading:  {cfg.train_path}")
    _log(f"[duckdb] writing: {cfg.out_path}")

    # Normalize input schema
    con.execute(
        f"""
        CREATE OR REPLACE VIEW train AS
        SELECT
            CAST(user_id AS BIGINT)    AS user_id,
            CAST(item_id AS BIGINT)    AS item_id,
            CAST(rating  AS DOUBLE)    AS rating,
            CAST(timestamp AS TIMESTAMP) AS ts
        FROM read_parquet('{cfg.train_path.as_posix()}');
        """
    )

    # Define as-of timestamp from TRAIN only (leakage-safe)
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
                    SUM(CASE WHEN t.ts >= (g.asof_ts - INTERVAL '{cfg.recent_days_7} days')  THEN 1 ELSE 0 END) AS n_last_7d,
                    SUM(CASE WHEN t.ts >= (g.asof_ts - INTERVAL '{cfg.recent_days_30} days') THEN 1 ELSE 0 END) AS n_last_30d
                FROM train t
                CROSS JOIN global_clock g
                GROUP BY t.item_id
            )
            SELECT
                p.item_id,
                CAST(p.n_interactions AS BIGINT) AS n_interactions,
                CAST(p.n_users AS BIGINT) AS n_users,
                CAST(p.rating_mean AS DOUBLE) AS rating_mean,
                CAST(COALESCE(p.rating_std, 0.0) AS DOUBLE) AS rating_std,
                CAST(p.first_ts AS TIMESTAMP) AS first_ts,
                CAST(p.last_ts AS TIMESTAMP) AS last_ts,
                CAST(r.days_since_last AS DOUBLE) AS days_since_last,
                CAST(COALESCE(rc.n_last_7d, 0) AS BIGINT) AS n_last_7d,
                CAST(COALESCE(rc.n_last_30d, 0) AS BIGINT) AS n_last_30d,
                CAST((COALESCE(rc.n_last_30d, 0) * 1.0) / NULLIF(p.n_interactions, 0) AS DOUBLE) AS recent_share_30d
            FROM per_item p
            LEFT JOIN recency r ON r.item_id = p.item_id
            LEFT JOIN recent_counts rc ON rc.item_id = p.item_id
        ) TO '{cfg.out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    n_items = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{cfg.out_path.as_posix()}');"
    ).fetchone()[0]

    # Meta (Netflix-style: explicit schema + reproducibility notes)
    meta: Dict[str, Any] = {
        "strategy": "train_only_item_features_with_asof_clock",
        "source_train_path": str(cfg.train_path),
        "out_path": str(cfg.out_path),
        "n_items": int(n_items),
        "windows_days": {"w7": int(cfg.recent_days_7), "w30": int(cfg.recent_days_30)},
        "leakage_safe": True,
        "asof_clock": "max(train.timestamp)",
        "columns": [
            "item_id",
            "n_interactions",
            "n_users",
            "rating_mean",
            "rating_std",
            "first_ts",
            "last_ts",
            "days_since_last",
            "n_last_7d",
            "n_last_30d",
            "recent_share_30d",
        ],
        "notes": "All item features computed strictly from TRAIN split. Recency features use as-of=max(train.ts).",
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(f"✅ Wrote: {cfg.out_path}")
    _log(f"✅ Meta : {cfg.meta_path}")


def main() -> None:
    # Entry-point expected by some tests and CLIs
    build_item_features(ItemFeatureConfig())


if __name__ == "__main__":
    main()
