# src/ntg/pipelines/build_dataset_duckdb.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class DuckDBBuildDatasetConfig:
    movielens_root: Path = Path("data/external/movielens/ml-32m")
    ratings_file: str = "ratings.csv"

    out_dir: Path = Path("data/processed")
    splits_dir: Path = Path("data/processed/splits")
    interactions_path: Path = Path("data/processed/interactions.parquet")
    metadata_path: Path = Path("data/processed/splits/metadata.json")

    # split config
    train_frac: float = 0.70
    val_frac: float = 0.15
    min_interactions: int = 5

    # write canonical interactions.parquet (optional; splits are the true deliverable)
    write_interactions: bool = True

    # DuckDB temp file (optional; helps if /tmp is small)
    duckdb_tmp_dir: Path = Path("data/interim/duckdb_tmp")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def run(cfg: DuckDBBuildDatasetConfig) -> None:
    if cfg.train_frac + cfg.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")
    if cfg.min_interactions < 1:
        raise ValueError("min_interactions must be >= 1")

    ratings_path = cfg.movielens_root / cfg.ratings_file
    _require_file(ratings_path)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    cfg.duckdb_tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[duckdb] reading: {ratings_path}", flush=True)
    print(f"[duckdb] writing splits to: {cfg.splits_dir}", flush=True)

    con = duckdb.connect(database=":memory:")
    # Make DuckDB more stable on large data
    con.execute(f"PRAGMA temp_directory='{cfg.duckdb_tmp_dir.as_posix()}';")
    con.execute("PRAGMA threads=4;")  # adjust to your CPU
    con.execute("PRAGMA enable_progress_bar=true;")

    # 1) Create canonical interactions table in DuckDB (no pandas)
    # MovieLens timestamp is seconds since epoch (UTC).
    con.execute(
        f"""
        CREATE OR REPLACE TABLE interactions AS
        SELECT
            CAST(userId AS INTEGER)  AS user_id,
            CAST(movieId AS INTEGER) AS item_id,
            CAST(rating AS REAL)     AS rating,
            CAST(to_timestamp(timestamp) AS TIMESTAMP) AS timestamp
        FROM read_csv_auto('{ratings_path.as_posix()}', header=true);
        """
    )

    # Optional: write a canonical interactions.parquet (compressed)
    if cfg.write_interactions:
        print(f"[duckdb] writing interactions parquet: {cfg.interactions_path}", flush=True)
        con.execute(
            f"""
            COPY (
                SELECT user_id, item_id, rating, timestamp
                FROM interactions
            ) TO '{cfg.interactions_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

    # 2) Create a view assigning split per row (per-user chronology)
    # Rule:
    # - if cnt < min_interactions => everything train
    # - else:
    #     train: rn <= train_end
    #     val:   train_end < rn <= val_end
    #     test:  rn > val_end
    #
    # train_end = max(1, floor(cnt*train_frac))
    # val_end   = max(train_end, floor(cnt*(train_frac+val_frac)))
    train_frac = cfg.train_frac
    val_frac = cfg.val_frac
    min_int = cfg.min_interactions

    con.execute(
        f"""
        CREATE OR REPLACE VIEW interactions_labeled AS
        WITH ranked AS (
            SELECT
                user_id,
                item_id,
                rating,
                timestamp,
                COUNT(*) OVER (PARTITION BY user_id) AS cnt,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp ASC, item_id ASC
                ) AS rn
            FROM interactions
        ),
        cutoffs AS (
            SELECT
                *,
                GREATEST(1, CAST(FLOOR(cnt * {train_frac}) AS BIGINT)) AS train_end,
                GREATEST(
                    GREATEST(1, CAST(FLOOR(cnt * {train_frac}) AS BIGINT)),
                    CAST(FLOOR(cnt * ({train_frac} + {val_frac})) AS BIGINT)
                ) AS val_end
            FROM ranked
        )
        SELECT
            user_id,
            item_id,
            rating,
            timestamp,
            CASE
                WHEN cnt < {min_int} THEN 'train'
                WHEN rn <= train_end THEN 'train'
                WHEN rn <= val_end THEN 'val'
                ELSE 'test'
            END AS split
        FROM cutoffs;
        """
    )

    # 3) Leakage sanity check (per user)
    # Ensures max(train_ts) <= min(val_ts) and max(val_ts) <= min(test_ts) when those splits exist.
    print("[duckdb] running leakage sanity checks...", flush=True)
    violations = con.execute(
        """
        WITH per_user AS (
            SELECT
                user_id,
                MAX(CASE WHEN split='train' THEN timestamp ELSE NULL END) AS train_max,
                MIN(CASE WHEN split='val'   THEN timestamp ELSE NULL END) AS val_min,
                MAX(CASE WHEN split='val'   THEN timestamp ELSE NULL END) AS val_max,
                MIN(CASE WHEN split='test'  THEN timestamp ELSE NULL END) AS test_min
            FROM interactions_labeled
            GROUP BY user_id
        )
        SELECT COUNT(*) AS n_violations
        FROM per_user
        WHERE
            (val_min IS NOT NULL AND train_max IS NOT NULL AND train_max > val_min)
            OR
            (test_min IS NOT NULL AND val_max IS NOT NULL AND val_max > test_min);
        """
    ).fetchone()[0]

    if violations != 0:
        raise RuntimeError(f"Leakage check failed: {violations} violating users found.")

    # 4) Write split parquet files (compressed)
    train_path = (cfg.splits_dir / "train.parquet").as_posix()
    val_path = (cfg.splits_dir / "val.parquet").as_posix()
    test_path = (cfg.splits_dir / "test.parquet").as_posix()

    print("[duckdb] writing train/val/test parquet...", flush=True)
    con.execute(
        f"""
        COPY (
            SELECT user_id, item_id, rating, timestamp
            FROM interactions_labeled
            WHERE split='train'
        ) TO '{train_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT user_id, item_id, rating, timestamp
            FROM interactions_labeled
            WHERE split='val'
        ) TO '{val_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT user_id, item_id, rating, timestamp
            FROM interactions_labeled
            WHERE split='test'
        ) TO '{test_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # 5) Metadata
    n_total = con.execute("SELECT COUNT(*) FROM interactions;").fetchone()[0]
    n_train = con.execute("SELECT COUNT(*) FROM interactions_labeled WHERE split='train';").fetchone()[0]
    n_val = con.execute("SELECT COUNT(*) FROM interactions_labeled WHERE split='val';").fetchone()[0]
    n_test = con.execute("SELECT COUNT(*) FROM interactions_labeled WHERE split='test';").fetchone()[0]

    meta = {
        "strategy": "duckdb_per_user_time",
        "n_total": float(n_total),
        "n_train": float(n_train),
        "n_val": float(n_val),
        "n_test": float(n_test),
        "frac_train": float(n_train / max(n_total, 1)),
        "frac_val": float(n_val / max(n_total, 1)),
        "frac_test": float(n_test / max(n_total, 1)),
        "train_frac": float(cfg.train_frac),
        "val_frac": float(cfg.val_frac),
        "min_interactions": float(cfg.min_interactions),
        "ratings_path": str(ratings_path),
        "interactions_parquet_written": bool(cfg.write_interactions),
    }

    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("✅ Day-1 complete (DuckDB).", flush=True)
    print(f" - splits: {cfg.splits_dir}", flush=True)
    print(f" - metadata: {cfg.metadata_path}", flush=True)
    if cfg.write_interactions:
        print(f" - interactions: {cfg.interactions_path}", flush=True)


if __name__ == "__main__":
    run(DuckDBBuildDatasetConfig())
