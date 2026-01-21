from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class InteractionFeatureConfig:
    train_path: Path = Path("data/processed/splits/train.parquet")
    out_dir: Path = Path("data/features")
    out_path: Path = Path("data/features/interaction_features_train.parquet")

    threads: int = 4
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def run(cfg: InteractionFeatureConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.train_path.exists():
        raise FileNotFoundError(f"Train split not found at {cfg.train_path}")

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

    # Per-user rating z-score + age_days
    con.execute(
        f"""
        COPY (
            WITH user_stats AS (
                SELECT
                    user_id,
                    AVG(rating) AS u_mean,
                    COALESCE(NULLIF(STDDEV_SAMP(rating), 0), 1.0) AS u_std
                FROM train
                GROUP BY user_id
            )
            SELECT
                t.user_id,
                t.item_id,
                t.rating,
                t.ts AS timestamp,
                (t.rating - s.u_mean) / s.u_std AS rating_z_user,
                DATE_DIFF('second', t.ts, g.asof_ts) / 86400.0 AS age_days
            FROM train t
            JOIN user_stats s USING(user_id)
            CROSS JOIN global_clock g
        ) TO '{cfg.out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    print(f"✅ Wrote: {cfg.out_path}")


if __name__ == "__main__":
    run(InteractionFeatureConfig())
