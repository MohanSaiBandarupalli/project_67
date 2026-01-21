from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb


def log_step(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass(frozen=True)
class ItemSimConfig:
    train_path: Path = Path("data/processed/splits/train.parquet")
    out_dir: Path = Path("outputs/graph")
    out_path: Path = Path("outputs/graph/item_item.parquet")
    meta_path: Path = Path("outputs/graph/item_item_meta.json")

    # Scale controls (16GB-safe, Netflix-grade)
    min_item_support: int = 200
    min_cooc: int = 20
    topk_per_item: int = 100
    min_cosine: float = 0.10

    # Power-user guardrails (CRITICAL to prevent quadratic explosion)
    min_items_per_user: int = 2
    max_items_per_user: int = 300

    # DuckDB tuning
    threads: int = 6
    memory_limit: str = "9GB"
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def build_item_similarity(cfg: ItemSimConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.train_path.exists():
        raise FileNotFoundError(f"Missing {cfg.train_path}. Run Day-1 first.")

    log_step("Connecting DuckDB (in-memory) + setting pragmas")
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")
    con.execute(f"PRAGMA memory_limit='{cfg.memory_limit}';")
    con.execute("PRAGMA enable_progress_bar=true;")
    con.execute("PRAGMA preserve_insertion_order=false;")

    log_step("Loading TRAIN interactions (leakage-safe)")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW train AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(item_id AS BIGINT) AS item_id
        FROM read_parquet('{cfg.train_path.as_posix()}');
        """
    )

    # 1) Item support
    log_step("Step 1/4: Computing item support + filtering rare items")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE item_support AS
        SELECT item_id, COUNT(*) AS item_cnt
        FROM train
        GROUP BY item_id
        HAVING COUNT(*) >= {cfg.min_item_support};
        """
    )
    n_items_supported = con.execute("SELECT COUNT(*) FROM item_support;").fetchone()[0]
    log_step(f"item_support ready: {n_items_supported:,} items retained")

    # 2) ui_raw distinct (supported items)
    log_step("Step 2/4: Building distinct user-item table (supported items only)")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ui_raw AS
        SELECT DISTINCT t.user_id, t.item_id
        FROM train t
        JOIN item_support s USING(item_id);
        """
    )

    # Power-user guardrails: keep only users with <= max_items_per_user items
    log_step("Applying power-user guardrails (min/max items per user)")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_item_counts AS
        SELECT user_id, COUNT(*) AS n_items
        FROM ui_raw
        GROUP BY user_id
        HAVING COUNT(*) >= {cfg.min_items_per_user}
           AND COUNT(*) <= {cfg.max_items_per_user};
        """
    )
    n_users_kept = con.execute("SELECT COUNT(*) FROM user_item_counts;").fetchone()[0]
    log_step(f"users kept after cap: {n_users_kept:,}")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ui AS
        SELECT u.user_id, u.item_id
        FROM ui_raw u
        JOIN user_item_counts c USING(user_id);
        """
    )
    n_ui = con.execute("SELECT COUNT(*) FROM ui;").fetchone()[0]
    log_step(f"ui ready: {n_ui:,} user-item pairs")

    # 3) co-occurrence (heavy join)
    log_step("Step 3/4: Computing co-occurrence pairs (heavy join)")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cooc AS
        SELECT
            a.item_id AS item_i,
            b.item_id AS item_j,
            COUNT(*) AS cooc_cnt
        FROM ui a
        JOIN ui b
          ON a.user_id = b.user_id
         AND a.item_id < b.item_id
        GROUP BY 1,2
        HAVING COUNT(*) >= {cfg.min_cooc};
        """
    )
    n_pairs = con.execute("SELECT COUNT(*) FROM cooc;").fetchone()[0]
    log_step(f"cooc ready: {n_pairs:,} item pairs")

    # 4) cosine similarity + directed edges
    log_step("Step 4/4: Computing cosine similarity + directed edges")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE edges_directed AS
        WITH sims AS (
            SELECT
                c.item_i,
                c.item_j,
                c.cooc_cnt,
                si.item_cnt AS cnt_i,
                sj.item_cnt AS cnt_j,
                (c.cooc_cnt * 1.0) / SQRT(si.item_cnt * sj.item_cnt) AS cosine
            FROM cooc c
            JOIN item_support si ON si.item_id = c.item_i
            JOIN item_support sj ON sj.item_id = c.item_j
        )
        SELECT item_i AS src_item, item_j AS dst_item, cosine, cooc_cnt, cnt_i AS src_cnt, cnt_j AS dst_cnt
        FROM sims
        WHERE cosine >= {cfg.min_cosine}
        UNION ALL
        SELECT item_j AS src_item, item_i AS dst_item, cosine, cooc_cnt, cnt_j AS src_cnt, cnt_i AS dst_cnt
        FROM sims
        WHERE cosine >= {cfg.min_cosine};
        """
    )
    n_edges_pre_topk = con.execute("SELECT COUNT(*) FROM edges_directed;").fetchone()[0]
    log_step(f"edges_directed ready: {n_edges_pre_topk:,} edges (pre-topK)")

    # topK per src_item
    log_step(f"Writing top-{cfg.topk_per_item} neighbors per item to parquet")
    con.execute(
        f"""
        COPY (
            SELECT src_item, dst_item, cosine, cooc_cnt, src_cnt, dst_cnt
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY src_item
                        ORDER BY cosine DESC, cooc_cnt DESC, dst_item ASC
                    ) AS rn
                FROM edges_directed
            )
            WHERE rn <= {cfg.topk_per_item}
        ) TO '{cfg.out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    n_items, n_edges = con.execute(
        f"""
        SELECT
            (SELECT COUNT(*) FROM item_support) AS n_items,
            (SELECT COUNT(*) FROM read_parquet('{cfg.out_path.as_posix()}')) AS n_edges
        """
    ).fetchone()

    meta = {
        "source_train": str(cfg.train_path),
        "n_items_supported": int(n_items),
        "n_users_kept": int(n_users_kept),
        "n_ui_pairs": int(n_ui),
        "n_pairs_cooc": int(n_pairs),
        "n_edges_pre_topk": int(n_edges_pre_topk),
        "n_edges_directed_topk": int(n_edges),
        "min_item_support": cfg.min_item_support,
        "min_cooc": cfg.min_cooc,
        "topk_per_item": cfg.topk_per_item,
        "min_cosine": cfg.min_cosine,
        "min_items_per_user": cfg.min_items_per_user,
        "max_items_per_user": cfg.max_items_per_user,
        "threads": cfg.threads,
        "memory_limit": cfg.memory_limit,
        "strategy": "user_cooccurrence_cosine_topk_guarded",
        "leakage_safe": True,
        "note": "TRAIN-only graph. Power-user cap prevents quadratic blow-ups. DuckDB spill enabled via temp_directory.",
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log_step(f"✅ Wrote: {cfg.out_path}")
    log_step(f"✅ Meta : {cfg.meta_path}")


if __name__ == "__main__":
    build_item_similarity(ItemSimConfig())
