from __future__ import annotations

import pytest
import pandas as pd


def test_graph_pipeline_imports():
    import ntg.graph.build_graph as m
    assert m is not None


def test_build_graph_smoke(sandbox, write_small_splits):
    """
    Runs Day-3 graph build if a callable main exists.
    Uses tiny data so it's fast.
    """
    import ntg.graph.build_graph as m

    run = getattr(m, "main", None)
    if run is None:
        pytest.skip("ntg.graph.build_graph.main not found")

    run()

    g = pd.read_parquet("outputs/graph/item_item.parquet")
    c = pd.read_parquet("outputs/graph/taste_clusters.parquet")

    assert {"src_item", "dst_item", "cosine"}.issubset(set(g.columns))
    assert g["cosine"].between(0, 1).all()
    assert {"item_id", "cluster_id"}.issubset(set(c.columns)) or len(c.columns) >= 2
