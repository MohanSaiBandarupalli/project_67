from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ntg.data.schemas import SCHEMA


@dataclass(frozen=True)
class MovieLensConfig:
    root_dir: Path
    ratings_file: str = "ratings.csv"

    # Write as parquet dataset first (streaming-friendly)
    interim_dataset_dir: Path = Path("data/interim/interactions_ds")

    # Optional compacted single file for convenience
    output_interactions_path: Path = Path("data/processed/interactions.parquet")

    chunksize: int = 2_000_000  # tune for your RAM (1M-3M typical)
    write_compacted: bool = True


class MovieLensIngestionError(RuntimeError):
    pass


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise MovieLensIngestionError(f"Required path not found: {path}")


def _optimize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast to reduce memory.
    user_id/item_id can fit in int32 for MovieLens.
    rating fits float32.
    timestamp -> datetime64[ns] (UTC stripped to naive).
    """
    df = df.rename(columns={"userId": SCHEMA.USER_ID, "movieId": SCHEMA.ITEM_ID})

    df[SCHEMA.USER_ID] = pd.to_numeric(df[SCHEMA.USER_ID], downcast="integer")
    df[SCHEMA.ITEM_ID] = pd.to_numeric(df[SCHEMA.ITEM_ID], downcast="integer")
    df[SCHEMA.RATING] = pd.to_numeric(df[SCHEMA.RATING], downcast="float")

    # Convert seconds->datetime, keep naive (consistent across pipeline)
    ts = pd.to_datetime(df[SCHEMA.TIMESTAMP], unit="s", utc=True)
    df[SCHEMA.TIMESTAMP] = ts.dt.tz_convert(None)

    return df[[SCHEMA.USER_ID, SCHEMA.ITEM_ID, SCHEMA.RATING, SCHEMA.TIMESTAMP]]


def build_interactions_dataset(
    movielens_root: Path,
    out_path: Optional[Path] = None,
    *,
    chunksize: Optional[int] = None,
    write_compacted: Optional[bool] = None,
) -> Path:
    """
    Streaming ingestion:
      ratings.csv -> parquet dataset (interim) -> optional compact parquet (processed)

    Returns the processed interactions parquet path if compacted,
    otherwise returns interim dataset dir.
    """
    cfg = MovieLensConfig(root_dir=movielens_root)
    if out_path is not None:
        cfg = MovieLensConfig(root_dir=movielens_root, output_interactions_path=out_path)
    if chunksize is not None:
        cfg = MovieLensConfig(
            root_dir=cfg.root_dir,
            ratings_file=cfg.ratings_file,
            interim_dataset_dir=cfg.interim_dataset_dir,
            output_interactions_path=cfg.output_interactions_path,
            chunksize=chunksize,
            write_compacted=cfg.write_compacted,
        )
    if write_compacted is not None:
        cfg = MovieLensConfig(
            root_dir=cfg.root_dir,
            ratings_file=cfg.ratings_file,
            interim_dataset_dir=cfg.interim_dataset_dir,
            output_interactions_path=cfg.output_interactions_path,
            chunksize=cfg.chunksize,
            write_compacted=write_compacted,
        )

    ratings_path = cfg.root_dir / cfg.ratings_file
    _ensure_exists(ratings_path)

    cfg.interim_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Clean old dataset shards (optional safety)
    # If you want to keep history, remove this block.
    for p in cfg.interim_dataset_dir.glob("part-*.parquet"):
        p.unlink()

    print(f"[ingestion] streaming read: {ratings_path}", flush=True)
    print(f"[ingestion] writing parquet dataset to: {cfg.interim_dataset_dir}", flush=True)

    reader = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        chunksize=cfg.chunksize,
    )

    total_rows = 0
    part = 0
    for chunk in reader:
        part += 1
        chunk = _optimize_chunk(chunk)
        total_rows += len(chunk)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        shard_path = cfg.interim_dataset_dir / f"part-{part:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")

        if part % 5 == 0:
            print(f"[ingestion] wrote {part} parts | rows so far: {total_rows:,}", flush=True)

    print(f"[ingestion] done. total rows: {total_rows:,}", flush=True)

    if not cfg.write_compacted:
        return cfg.interim_dataset_dir

    # Compact to a single parquet (still can be big, but parquet read is cheaper than CSV)
    cfg.output_interactions_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ingestion] compacting → {cfg.output_interactions_path}", flush=True)

    dataset = ds.dataset(str(cfg.interim_dataset_dir), format="parquet")
    table = dataset.to_table()  # this loads all into memory; if this OOMs, skip compaction.
    pq.write_table(table, cfg.output_interactions_path, compression="zstd")

    print(f"[ingestion] compacted interactions saved.", flush=True)
    return cfg.output_interactions_path


# =============================================================================
# Split exports (required by tests via ntg.data.ingestion._import_splits())
# Single source of truth: ntg.data.splits
# =============================================================================
from ntg.data.splits import (  # noqa: E402
    GlobalTimeSplitConfig,
    PerUserTimeSplitConfig,
    TimeSplitConfig,
    time_split_global,
    time_split_per_user,
)

__all__ = [
    "MovieLensConfig",
    "MovieLensIngestionError",
    "build_interactions_dataset",
    # split API
    "TimeSplitConfig",
    "PerUserTimeSplitConfig",
    "GlobalTimeSplitConfig",
    "time_split_global",
    "time_split_per_user",
]
