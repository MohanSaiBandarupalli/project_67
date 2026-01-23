# src/ntg/models/ranker.py
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
    # Inputs (leakage-safe)
    train_path: Path = Path("data/processed/splits/train.parquet")
    val_path: Path = Path("data/processed/splits/val.parquet")

    # Optional input (graph). If missing -> popularity-only fallback.
    graph_path: Path = Path("outputs/graph/item_item.parquet")

    # Outputs
    out_topk_path: Path = Path("outputs/recommendations/topk.parquet")
    out_metrics_path: Path = Path("reports/ranking_metrics.json")

    # Ranking params
    k: int = 50
    candidates_per_user: int = 300

    # Guardrails
    min_rating: float = 4.0
    max_hist_items_per_user: int = 200
    min_user_hist: int = 5

    # Score weights
    w_graph: float = 1.0
    w_pop: float = 0.15

    # DuckDB runtime
    threads: int = 4
    memory_limit: str = "6GB"
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def build_topk_and_eval(cfg: RankerConfig) -> None:
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_topk_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Required inputs
    for p in [cfg.train_path, cfg.val_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    # Optional graph (do NOT fail tests if missing)
    graph_available = cfg.graph_path.exists()

    log("Connecting DuckDB (in-memory)")
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA memory_limit='{cfg.memory_limit}';")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")
    con.execute("PRAGMA enable_progress_bar=true;")
    con.execute("PRAGMA preserve_insertion_order=false;")

    log("Loading TRAIN + VAL")
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

    if graph_available:
        log("Loading graph")
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
    else:
        # Deterministic fallback: empty graph
        log("Graph missing -> running popularity-only fallback (deterministic)")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE graph (
                src_item BIGINT,
                dst_item BIGINT,
                cosine   DOUBLE
            );
            """
        )
        con.execute("CREATE OR REPLACE VIEW graph AS SELECT * FROM graph;")

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

    # Deterministic user history (TRAIN-only, positive-only, capped)
    # IMPORTANT: NO RANDOM() -> determinism for tests
    log("Building user history (TRAIN-only, positive-only, capped, deterministic)")
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
                    ORDER BY rating DESC, item_id ASC
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

    # Ensure we cover users even if they have 0 positives (so we can still recommend)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE all_users AS
        SELECT DISTINCT user_id FROM train
        UNION
        SELECT DISTINCT user_id FROM val;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE eligible AS
        SELECT
            u.user_id,
            COALESCE(h.n_hist, 0) AS n_hist
        FROM all_users u
        LEFT JOIN user_hist_cnt h USING(user_id);
        """
    )

    # Popularity fallback list
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

    # Candidate generation only if graph exists (or graph table might be empty)
    if graph_available:
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
    else:
        # No graph -> empty candidate pool
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE cand_pool (
                user_id BIGINT,
                item_id BIGINT,
                score   DOUBLE
            );
            """
        )

    # Build Top-K recommendations:
    # - If graph missing -> always popularity fallback
    # - Else: popularity fallback for low-history, candidate-based for others
    log("Building Top-K recommendations")
    if not graph_available:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE topk AS
            SELECT
                e.user_id,
                p.item_id,
                p.pop_log AS score,
                ROW_NUMBER() OVER (
                    PARTITION BY e.user_id
                    ORDER BY p.pop_log DESC, p.item_id ASC
                ) AS rank
            FROM eligible e
            JOIN pop_fallback p ON TRUE;
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE topk AS
            WITH pool_or_pop AS (
                -- popularity fallback for low-history users
                SELECT
                    e.user_id,
                    p.item_id AS item_id,
                    p.pop_log AS score
                FROM eligible e
                JOIN pop_fallback p ON TRUE
                WHERE e.n_hist < {cfg.min_user_hist}

                UNION ALL

                -- candidate-based scoring for sufficient-history users
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

    # Offline eval on VAL (requires your metrics.py to exist)
    log("Evaluating on VAL")
    topk_df = pd.read_parquet(cfg.out_topk_path)
    gt_val = pd.read_parquet(cfg.val_path)[["user_id", "item_id"]].copy()

    from ntg.evaluation.metrics import RankingMetricsConfig, evaluate_topk

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
        "graph_available": graph_available,
        "duckdb": {
            "threads": cfg.threads,
            "memory_limit": cfg.memory_limit,
            "tmp_dir": str(cfg.tmp_dir),
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Leakage-safe: TRAIN-only history; popularity fallback always available. "
            "Deterministic ordering (no RANDOM)."
        ),
    }

    cfg.out_metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"✅ Wrote: {cfg.out_topk_path}")
    log(f"✅ Wrote: {cfg.out_metrics_path}")


def main() -> None:
    build_topk_and_eval(RankerConfig())


if __name__ == "__main__":
    main()
