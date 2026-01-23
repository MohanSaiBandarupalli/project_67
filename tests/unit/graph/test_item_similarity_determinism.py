from __future__ import annotations

import pytest
import pandas as pd


def test_graph_build_deterministic_on_tiny_data(sandbox, write_small_splits):
    """
    Runs graph build twice and checks identical output when deterministic.
    If your build_graph has randomness, you can seed or skip.
    """
    import ntg.graph.build_graph as m

    run = getattr(m, "main", None)
    if run is None:
        pytest.skip("ntg.graph.build_graph.main not found")

    run()
    a = pd.read_parquet("outputs/graph/item_item.parquet").sort_values(["src_item", "dst_item"]).reset_index(drop=True)

    # cleanup outputs and run again
    import os
    os.remove("outputs/graph/item_item.parquet")
    os.remove("outputs/graph/taste_clusters.parquet")

    run()
    b = pd.read_parquet("outputs/graph/item_item.parquet").sort_values(["src_item", "dst_item"]).reset_index(drop=True)

    assert a.equals(b)
