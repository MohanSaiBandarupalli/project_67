# src/ntg/features/build_interaction_features.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb


@dataclass(frozen=True)
class InteractionFeatureConfig:
    """
    Leakage-safe interaction feature builder (TRAIN only).

    Produces one row per (user_id, item_id, timestamp) interaction in TRAIN with:
      - rating_z_user : user-normalized rating
      - age_days      : how old the interaction is relative to asof_ts (max TRAIN ts)

    Outputs:
      - data/features/interaction_features_train.parquet
      - data/features/interaction_features_train_meta.json
    """

    train_path: Path = Path("data/processed/splits/train.parquet")

    out_dir: Path = Path("data/features")
    out_path: Path = Path("data/features/interaction_features_train.parquet")
    meta_path: Path = Path("data/features/interaction_features_train_meta.json")

    threads: int = 4
    tmp_dir: Path = Path("data/interim/duckdb_tmp")
    enable_progress_bar: bool = False


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_interaction_features(cfg: Optional[InteractionFeatureConfig] = None) -> Path:
    """
    Public API expected by unit tests (fallback path).
    Returns the written parquet path.
    """
    cfg = cfg or InteractionFeatureConfig()
    run(cfg)
    return cfg.out_path


def run(cfg: InteractionFeatureConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.train_path.exists():
        raise FileNotFoundError(f"Train split not found at {cfg.train_path}")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")
    if cfg.enable_progress_bar:
        con.execute("PRAGMA enable_progress_bar=true;")

    _log("=== Day-2: Interaction features (train-only) ===")
    _log(f"[duckdb] reading : {cfg.train_path}")
    _log(f"[duckdb] writing : {cfg.out_path}")

    # Canonicalize schema from parquet -> stable types
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

    # Global "as-of" clock computed from TRAIN only (leakage-safe)
    con.execute("CREATE OR REPLACE VIEW global_clock AS SELECT MAX(ts) AS asof_ts FROM train;")

    # Feature build
    # - user_stats: mean + std of ratings per user (std guard for constant users)
    # - rating_z_user: z-score
    # - age_days: how old the event is relative to asof_ts
    con.execute(
        f"""
        COPY (
            WITH user_stats AS (
                SELECT
                    user_id,
                    AVG(rating) AS u_mean,
                    CASE
                        WHEN STDDEV_SAMP(rating) IS NULL THEN 1.0
                        WHEN STDDEV_SAMP(rating) = 0 THEN 1.0
                        ELSE STDDEV_SAMP(rating)
                    END AS u_std
                FROM train
                GROUP BY user_id
            )
            SELECT
                t.user_id,
                t.item_id,
                CAST(t.rating AS DOUBLE) AS rating,
                CAST(t.ts AS TIMESTAMP) AS timestamp,
                CAST((t.rating - s.u_mean) / s.u_std AS DOUBLE) AS rating_z_user,
                CAST(DATE_DIFF('second', t.ts, g.asof_ts) / 86400.0 AS DOUBLE) AS age_days
            FROM train t
            JOIN user_stats s USING(user_id)
            CROSS JOIN global_clock g
        ) TO '{cfg.out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # Meta for reproducibility/contracts
    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{cfg.out_path.as_posix()}');"
    ).fetchone()[0]

    # Column list is part of the contract (helps tests + docs)
    meta: Dict[str, Any] = {
        "strategy": "train_only_interaction_features",
        "source_train_path": str(cfg.train_path),
        "out_path": str(cfg.out_path),
        "n_rows": int(n_rows),
        "leakage_safe": True,
        "asof_clock": "max(train.timestamp)",
        "features": {
            "rating_z_user": "Z-score of rating per user (mean/std from TRAIN only; std guarded).",
            "age_days": "Days since interaction relative to TRAIN asof_ts.",
        },
        "columns": ["user_id", "item_id", "rating", "timestamp", "rating_z_user", "age_days"],
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(f"✅ Wrote: {cfg.out_path}")
    _log(f"✅ Meta : {cfg.meta_path}")


def main() -> None:
    # Preferred entrypoint for tests/CLI
    build_interaction_features(InteractionFeatureConfig())


if __name__ == "__main__":
    main()
