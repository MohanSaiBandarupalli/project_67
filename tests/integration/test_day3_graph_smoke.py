from __future__ import annotations

from pathlib import Path
import pytest


@pytest.mark.integration
def test_day3_graph_smoke(has_movielens_data):
    if not has_movielens_data:
        pytest.skip("MovieLens data not present")

    import ntg.graph.build_graph as g
    g.main()

    assert Path("outputs/graph/item_item.parquet").exists()
    assert Path("outputs/graph/taste_clusters.parquet").exists()
