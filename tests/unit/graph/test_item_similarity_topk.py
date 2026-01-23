from __future__ import annotations

import pytest
import pandas as pd


def test_item_item_topk_property_if_output_exists(project_root):
    """
    If outputs exist locally, check topK property.
    (CI won't have outputs, so we skip unless present.)
    """
    p = project_root / "outputs/graph/item_item.parquet"
    if not p.exists():
        pytest.skip("no local item_item.parquet present")
    g = pd.read_parquet(p)

    assert "src_item" in g.columns
    assert "dst_item" in g.columns
    assert "cosine" in g.columns

    # should be <= topK per item (your default is 100)
    max_k = g.groupby("src_item").size().max()
    assert max_k <= 200  # generous; keep stable even if you tune topK
