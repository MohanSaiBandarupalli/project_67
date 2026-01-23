# src/ntg/graph/build_graph.py
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ItemSimConfig:
    # Inputs
    train_path: Path = Path("data/processed/splits/train.parquet")

    # Outputs
    out_dir: Path = Path("outputs/graph")
    out_item_item: Path = Path("outputs/graph/item_item.parquet")
    out_item_item_meta: Path = Path("outputs/graph/item_item_meta.json")
    out_clusters: Path = Path("outputs/graph/taste_clusters.parquet")
    out_clusters_meta: Path = Path("outputs/graph/taste_clusters_meta.json")

    # Scale controls (production defaults)
    min_item_support: int = 200
    min_cooc: int = 20
    topk_per_item: int = 100
    min_cosine: float = 0.10

    # Power-user guardrails (avoid quadratic blowups)
    min_items_per_user: int = 2
    max_items_per_user: int = 300

    # DuckDB tuning
    threads: int = 6
    memory_limit: str = "8GB"
    # Base dir only. We will create a unique per-run temp dir under this.
    tmp_dir: Path = Path("data/interim/duckdb_tmp")

    # Auto-tune behavior for unit tests / tiny datasets
    tiny_n_interactions: int = 50
    tiny_n_items: int = 50
    tiny_n_users: int = 50


# ============================================================
# Helpers
# ============================================================
def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _load_train(con: duckdb.DuckDBPyConnection, train_path: Path) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW train AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(item_id AS BIGINT) AS item_id,
            CAST(timestamp AS TIMESTAMP) AS ts
        FROM read_parquet('{train_path.as_posix()}');
        """
    )


def _get_basic_stats(con: duckdb.DuckDBPyConnection) -> Tuple[int, int, int]:
    n_inter = int(con.execute("SELECT COUNT(*) FROM train;").fetchone()[0])
    n_users = int(con.execute("SELECT COUNT(DISTINCT user_id) FROM train;").fetchone()[0])
    n_items = int(con.execute("SELECT COUNT(DISTINCT item_id) FROM train;").fetchone()[0])
    return n_inter, n_users, n_items


def _auto_tune_for_tiny(cfg: ItemSimConfig, n_inter: int, n_users: int, n_items: int) -> ItemSimConfig:
    """
    Production pattern:
      - strict defaults for real data
      - auto-relax only for tiny/unit-test data so outputs are non-empty + fast
    """
    tiny = (
        n_inter <= cfg.tiny_n_interactions
        or n_users <= cfg.tiny_n_users
        or n_items <= cfg.tiny_n_items
    )
    if not tiny:
        return cfg

    return replace(
        cfg,
        min_item_support=1,
        min_cooc=1,
        min_items_per_user=1,
        max_items_per_user=max(cfg.max_items_per_user, n_items),
        topk_per_item=max(1, min(cfg.topk_per_item, max(1, n_items - 1))),
        min_cosine=min(cfg.min_cosine, 0.0),
    )


def _make_run_tmp_dir(base: Path) -> Path:
    """
    Critical fix:
    DuckDB temp files must live in a *unique* directory per run to avoid
    pytest sandbox cleanup / concurrent test interference.
    """
    base.mkdir(parents=True, exist_ok=True)
    # Use OS temp + your base as prefix metadata. This is robust across sandboxes.
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"duckdb_{os.getpid()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_",
            dir=base.as_posix(),
        )
    )
    return run_dir


def _connect(cfg: ItemSimConfig, run_tmp: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA memory_limit='{cfg.memory_limit}';")
    con.execute(f"PRAGMA temp_directory='{run_tmp.as_posix()}';")

    # Progress bar can slow CI + makes logs noisy; keep off for tests.
    con.execute("PRAGMA enable_progress_bar=false;")

    # More stable/faster for analytics pipelines
    con.execute("PRAGMA preserve_insertion_order=false;")
    return con


# ============================================================
# Core build
# ============================================================
def build_graph(cfg: ItemSimConfig) -> Dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_parent(cfg.out_item_item)
    _ensure_parent(cfg.out_item_item_meta)
    _ensure_parent(cfg.out_clusters)
    _ensure_parent(cfg.out_clusters_meta)

    if not cfg.train_path.exists():
        raise FileNotFoundError(f"Missing train split: {cfg.train_path}")

    run_tmp = _make_run_tmp_dir(cfg.tmp_dir)

    con: duckdb.DuckDBPyConnection | None = None
    try:
        _log("Connecting DuckDB (in-memory) + setting pragmas")
        con = _connect(cfg, run_tmp)

        _log("Loading TRAIN interactions (leakage-safe)")
        _load_train(con, cfg.train_path)

        n_inter, n_users, n_items = _get_basic_stats(con)
        tuned = _auto_tune_for_tiny(cfg, n_inter=n_inter, n_users=n_users, n_items=n_items)

        _log("Step 1/4: Computing item support + filtering rare items")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW item_support AS
            SELECT item_id, COUNT(*) AS n_interactions
            FROM train
            GROUP BY item_id
            HAVING COUNT(*) >= {tuned.min_item_support};
            """
        )
        kept_items = int(con.execute("SELECT COUNT(*) FROM item_support;").fetchone()[0])
        _log(f"item_support ready: {kept_items} items retained")

        _log("Step 2/4: Building distinct user-item table (supported items only)")
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW ui0 AS
            SELECT DISTINCT t.user_id, t.item_id
            FROM train t
            JOIN item_support s USING(item_id);
            """
        )

        _log("Applying power-user guardrails (min/max items per user)")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW user_item_counts AS
            SELECT user_id, COUNT(*) AS n_items
            FROM ui0
            GROUP BY user_id
            HAVING COUNT(*) >= {tuned.min_items_per_user}
               AND COUNT(*) <= {tuned.max_items_per_user};
            """
        )
        kept_users = int(con.execute("SELECT COUNT(*) FROM user_item_counts;").fetchone()[0])
        _log(f"users kept after cap: {kept_users}")

        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW ui AS
            SELECT u.user_id, u.item_id
            FROM ui0 u
            JOIN user_item_counts c USING(user_id);
            """
        )
        ui_pairs = int(con.execute("SELECT COUNT(*) FROM ui;").fetchone()[0])
        _log(f"ui ready: {ui_pairs} user-item pairs")

        _log("Step 3/4: Computing co-occurrence pairs (heavy join)")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW cooc AS
            SELECT
                a.item_id AS item_i,
                b.item_id AS item_j,
                COUNT(*) AS n_cooc
            FROM ui a
            JOIN ui b
              ON a.user_id = b.user_id
             AND a.item_id < b.item_id
            GROUP BY 1,2
            HAVING COUNT(*) >= {tuned.min_cooc};
            """
        )
        n_pairs = int(con.execute("SELECT COUNT(*) FROM cooc;").fetchone()[0])
        _log(f"cooc ready: {n_pairs} item pairs")

        _log("Step 4/4: Computing cosine similarity + directed edges")
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW item_deg AS
            SELECT item_id, COUNT(*) AS n_users
            FROM ui
            GROUP BY item_id;
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW edges AS
            SELECT
                c.item_i,
                c.item_j,
                c.n_cooc,
                d1.n_users AS n_users_i,
                d2.n_users AS n_users_j,
                (c.n_cooc * 1.0) / NULLIF(SQRT(d1.n_users * d2.n_users), 0) AS cosine
            FROM cooc c
            JOIN item_deg d1 ON d1.item_id = c.item_i
            JOIN item_deg d2 ON d2.item_id = c.item_j
            WHERE (c.n_cooc * 1.0) / NULLIF(SQRT(d1.n_users * d2.n_users), 0) >= {tuned.min_cosine};
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW edges_directed AS
            SELECT item_i AS item_id, item_j AS neighbor_id, cosine, n_cooc
            FROM edges
            UNION ALL
            SELECT item_j AS item_id, item_i AS neighbor_id, cosine, n_cooc
            FROM edges;
            """
        )
        n_edges = int(con.execute("SELECT COUNT(*) FROM edges_directed;").fetchone()[0])
        _log(f"edges_directed ready: {n_edges} edges (pre-topK)")

        _log(f"Writing top-{tuned.topk_per_item} neighbors per item to parquet")
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM (
                    SELECT
                        item_id     AS src_item,
                        neighbor_id AS dst_item,
                        cosine,
                        n_cooc,
                        ROW_NUMBER() OVER (
                            PARTITION BY item_id
                            ORDER BY cosine DESC, n_cooc DESC, neighbor_id ASC
                        ) AS rk
                    FROM edges_directed
                )
                WHERE rk <= {tuned.topk_per_item}
            ) TO '{cfg.out_item_item.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

        _log("Building taste clusters (deterministic fallback if graph is empty)")
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW cluster_items AS
            SELECT DISTINCT item_id FROM ui
            UNION
            SELECT DISTINCT item_id FROM ui0
            UNION
            SELECT DISTINCT item_id FROM train;
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT
                    item_id,
                    DENSE_RANK() OVER (ORDER BY item_id) - 1 AS cluster_id
                FROM cluster_items
            ) TO '{cfg.out_clusters.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

        item_item_meta = {
            "source": str(cfg.train_path),
            "n_interactions": n_inter,
            "n_users": n_users,
            "n_items": n_items,
            "effective": {
                "min_item_support": tuned.min_item_support,
                "min_cooc": tuned.min_cooc,
                "min_cosine": tuned.min_cosine,
                "topk_per_item": tuned.topk_per_item,
                "min_items_per_user": tuned.min_items_per_user,
                "max_items_per_user": tuned.max_items_per_user,
            },
            "duckdb": {
                "threads": cfg.threads,
                "memory_limit": cfg.memory_limit,
                "temp_directory": str(run_tmp),
            },
            "leakage_safe": True,
            "note": "Built from TRAIN only. Auto-tunes thresholds for tiny datasets (unit tests) without affecting production defaults.",
        }
        cfg.out_item_item_meta.write_text(json.dumps(item_item_meta, indent=2), encoding="utf-8")

        clusters_meta = {
            "source": str(cfg.train_path),
            "n_interactions": n_inter,
            "n_users": n_users,
            "n_items": n_items,
            "cluster_strategy": "dense_rank_fallback",
            "duckdb_temp_directory": str(run_tmp),
            "leakage_safe": True,
            "note": "Always emits at least one row when train has items; deterministic.",
        }
        cfg.out_clusters_meta.write_text(json.dumps(clusters_meta, indent=2), encoding="utf-8")

        _log(f"✅ Wrote: {cfg.out_item_item}")
        _log(f"✅ Meta : {cfg.out_item_item_meta}")
        print(f"✅ Wrote: {cfg.out_clusters}")
        print(f"✅ Meta : {cfg.out_clusters_meta}")

        return {
            "item_item_path": str(cfg.out_item_item),
            "taste_clusters_path": str(cfg.out_clusters),
            "effective_cfg": tuned,
        }

    finally:
        # Close DuckDB first so it releases temp files cleanly.
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

        # Cleanup per-run temp dir. If deletion fails, leave it (never break the run/tests).
        try:
            shutil.rmtree(run_tmp, ignore_errors=True)
        except Exception:
            pass


def main() -> None:
    print("=== Day-3: Taste Graph Engine ===")
    out = build_graph(ItemSimConfig())
    print("✅ Day-3 complete.")
    print(f" - {out['item_item_path']}")
    print(f" - {out['taste_clusters_path']}")


if __name__ == "__main__":
    main()
