from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass(frozen=True)
class RankerConfig:
    """
    Candidate-based ranker for offline evaluation.

    Netflix-style: two-stage mindset
      (1) candidate generation (graph neighbors, TRAIN-only)
      (2) ranking (simple linear score)
      (3) offline eval on VAL

    Guardrails are critical to prevent blow-ups on ML-32M:
      - cap history per user
      - use positive-only interactions for candidate generation
      - cap candidate pool per user
      - avoid expensive "anti-join after explosion"
    """
    # Inputs (leakage-safe)
    train_path: Path = Path("data/processed/splits/train.parquet")
    val_path: Path = Path("data/processed/splits/val.parquet")
    test_path: Path = Path("data/processed/splits/test.parquet")

    graph_path: Path = Path("outputs/graph/item_item.parquet")

    # Outputs
    out_topk_path: Path = Path("outputs/recommendations/topk.parquet")
    out_metrics_path: Path = Path("reports/ranking_metrics.json")

    # Ranking params
    k: int = 50
    candidates_per_user: int = 300

    # Guardrails (the fix for your 44% stall)
    min_rating: float = 4.0              # MovieLens: treat >=4 as positive signal
    max_hist_items_per_user: int = 200   # cap user history used for candidates
    min_user_hist: int = 5               # below this -> popularity fallback only

    # Score weights
    w_graph: float = 1.0
    w_pop: float = 0.15

    # DuckDB runtime (WSL-friendly)
    threads: int = 4
    memory_limit: str = "6GB"
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def build_topk_and_eval(cfg: RankerConfig) -> None:
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_topk_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    for p in [cfg.train_path, cfg.val_path, cfg.graph_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    log("Connecting DuckDB (in-memory)")
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA memory_limit='{cfg.memory_limit}';")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")
    con.execute("PRAGMA enable_progress_bar=true;")
    con.execute("PRAGMA preserve_insertion_order=false;")

    log("Loading TRAIN + VAL + graph")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW train AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(item_id AS BIGINT) AS item_id,
            CAST(rating AS DOUBLE)  AS rating
        FROM read_parquet('{cfg.train_path.as_posix()}');
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW val AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(item_id AS BIGINT) AS item_id
        FROM read_parquet('{cfg.val_path.as_posix()}');
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW graph AS
        SELECT
            CAST(src_item AS BIGINT) AS src_item,
            CAST(dst_item AS BIGINT) AS dst_item,
            CAST(cosine AS DOUBLE)   AS cosine
        FROM read_parquet('{cfg.graph_path.as_posix()}');
        """
    )

    # Popularity prior (TRAIN-only)
    log("Computing item popularity prior (TRAIN-only)")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE item_pop AS
        SELECT item_id, COUNT(*) AS pop_cnt
        FROM train
        GROUP BY item_id;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE item_pop_norm AS
        SELECT item_id, LOG(1 + pop_cnt) AS pop_log
        FROM item_pop;
        """
    )

    # Build guarded user history (TRAIN-only, positive-only, capped)
    log("Building user history (TRAIN-only, positive-only, capped)")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_hist AS
        SELECT user_id, item_id
        FROM (
            SELECT
                user_id,
                item_id,
                rating,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY rating DESC, RANDOM()
                ) AS rn
            FROM train
            WHERE rating >= {cfg.min_rating}
        )
        WHERE rn <= {cfg.max_hist_items_per_user};
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE user_hist_cnt AS
        SELECT user_id, COUNT(*) AS n_hist
        FROM user_hist
        GROUP BY user_id;
        """
    )

    n_users = con.execute("SELECT COUNT(*) FROM user_hist_cnt").fetchone()[0]
    log(f"user_hist users: {n_users:,}")

    # Popularity fallback list (for cold/low-history users)
    log("Preparing popularity fallback list")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pop_fallback AS
        SELECT item_id, pop_log
        FROM item_pop_norm
        ORDER BY pop_log DESC, item_id ASC
        LIMIT {cfg.candidates_per_user};
        """
    )

    # Candidate generation + filtering already-seen (CHEAP: filter inside CTE)
    # This avoids the expensive anti-join on a huge exploded cand table.
    log("Generating candidates from graph neighbors (excluding already-seen)")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cand_scored AS
        WITH neigh AS (
            SELECT
                h.user_id,
                g.dst_item AS cand_item,
                SUM(g.cosine) AS graph_mass
            FROM user_hist h
            JOIN graph g
              ON h.item_id = g.src_item
            GROUP BY 1,2
        ),
        filtered AS (
            SELECT n.*
            FROM neigh n
            WHERE NOT EXISTS (
                SELECT 1
                FROM user_hist h
                WHERE h.user_id = n.user_id AND h.item_id = n.cand_item
            )
        ),
        with_pop AS (
            SELECT
                f.user_id,
                f.cand_item,
                f.graph_mass,
                COALESCE(p.pop_log, 0.0) AS pop_log,
                ({cfg.w_graph} * f.graph_mass + {cfg.w_pop} * COALESCE(p.pop_log, 0.0)) AS score
            FROM filtered f
            LEFT JOIN item_pop_norm p
              ON p.item_id = f.cand_item
        )
        SELECT * FROM with_pop;
        """
    )

    n_cand = con.execute("SELECT COUNT(*) FROM cand_scored").fetchone()[0]
    log(f"candidate rows (pre-cap): {n_cand:,}")

    # Cap candidates per user
    log("Capping candidates per user")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cand_pool AS
        SELECT user_id, cand_item AS item_id, score
        FROM (
            SELECT
                user_id,
                cand_item,
                score,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY score DESC, cand_item ASC
                ) AS rn
            FROM cand_scored
        )
        WHERE rn <= {cfg.candidates_per_user};
        """
    )

    # Build Top-K:
    # - low-history users: popularity list
    # - normal users: candidate pool
    log("Building Top-K recommendations")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE topk AS
        WITH eligible AS (
            SELECT user_id, n_hist
            FROM user_hist_cnt
        ),
        pool_or_pop AS (
            -- popularity fallback
            SELECT
                e.user_id,
                p.item_id AS item_id,
                p.pop_log AS score
            FROM eligible e
            JOIN pop_fallback p ON TRUE
            WHERE e.n_hist < {cfg.min_user_hist}

            UNION ALL

            -- candidate-based scoring
            SELECT
                e.user_id,
                c.item_id,
                c.score
            FROM eligible e
            JOIN cand_pool c
              ON e.user_id = c.user_id
            WHERE e.n_hist >= {cfg.min_user_hist}
        )
        SELECT
            user_id,
            item_id,
            score,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY score DESC, item_id ASC
            ) AS rank
        FROM pool_or_pop;
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE topk_k AS
        SELECT user_id, item_id, score, rank
        FROM topk
        WHERE rank <= {cfg.k};
        """
    )

    n_recs = con.execute("SELECT COUNT(*) FROM topk_k").fetchone()[0]
    log(f"total recommendations written: {n_recs:,}")

    # Write parquet
    log(f"Writing: {cfg.out_topk_path}")
    con.execute(
        f"""
        COPY (
            SELECT user_id, item_id, score, rank
            FROM topk_k
            ORDER BY user_id ASC, rank ASC
        ) TO '{cfg.out_topk_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # Offline eval on VAL
    log("Evaluating on VAL")
    topk_df = pd.read_parquet(cfg.out_topk_path)
    gt_val = pd.read_parquet(cfg.val_path)[["user_id", "item_id"]].copy()

    from ntg.utils.metrics import RankingMetricsConfig, evaluate_topk

    metrics = evaluate_topk(
        topk=topk_df,
        ground_truth=gt_val,
        cfg=RankingMetricsConfig(k_list=(5, 10, 20, cfg.k)),
    )

    report = {
        "split": "val",
        "k": cfg.k,
        "candidates_per_user": cfg.candidates_per_user,
        "min_rating": cfg.min_rating,
        "max_hist_items_per_user": cfg.max_hist_items_per_user,
        "min_user_hist": cfg.min_user_hist,
        "weights": {"w_graph": cfg.w_graph, "w_pop": cfg.w_pop},
        "duckdb": {"threads": cfg.threads, "memory_limit": cfg.memory_limit, "tmp_dir": str(cfg.tmp_dir)},
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": "Leakage-safe: TRAIN-only positive history; graph candidates; popularity fallback for low-history users. Guardrails prevent candidate blow-ups.",
    }

    cfg.out_metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"✅ Wrote: {cfg.out_topk_path}")
    log(f"✅ Wrote: {cfg.out_metrics_path}")


if __name__ == "__main__":
    build_topk_and_eval(RankerConfig())
