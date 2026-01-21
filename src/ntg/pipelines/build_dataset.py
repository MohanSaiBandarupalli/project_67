# src/ntg/pipelines/build_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ntg.data.ingestion import build_interactions_dataset
from ntg.data.splits import TimeSplitConfig, time_split_per_user
from ntg.data.validation import validate_interactions_df, validate_per_user_order


@dataclass(frozen=True)
class BuildDatasetConfig:
    movielens_root: Path = Path("data/external/movielens/ml-32m")
    interactions_out: Path = Path("data/processed/interactions.parquet")
    splits_dir: Path = Path("data/processed/splits")
    metadata_path: Path = Path("data/processed/splits/metadata.json")

    split: TimeSplitConfig = TimeSplitConfig(train_frac=0.70, val_frac=0.15, min_interactions=5)


def _read_interactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Interactions not found: {path}")
    return pd.read_parquet(path)


def run(cfg: BuildDatasetConfig) -> None:
    # 1) Ingest canonical interactions
    out_path = build_interactions_dataset(cfg.movielens_root, out_path=cfg.interactions_out)

    df = _read_interactions(out_path)
    validate_interactions_df(df)

    # 2) Split
    train, val, test, meta = time_split_per_user(df, cfg.split)

    # 3) Validate leakage
    validate_per_user_order(train, val, test)

    # 4) Write outputs
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cfg.splits_dir / "train.parquet", index=False)
    val.to_parquet(cfg.splits_dir / "val.parquet", index=False)
    test.to_parquet(cfg.splits_dir / "test.parquet", index=False)

    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("✅ Day-1 complete")
    print(f" - interactions: {cfg.interactions_out}")
    print(f" - train/val/test: {cfg.splits_dir}")
    print(f" - metadata: {cfg.metadata_path}")


if __name__ == "__main__":
    run(BuildDatasetConfig())
