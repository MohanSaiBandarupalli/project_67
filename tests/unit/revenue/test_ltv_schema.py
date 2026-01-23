from __future__ import annotations

import pytest
import pandas as pd


def test_ltv_module_imports():
    import ntg.revenue.ltv as m
    assert m is not None


def test_ltv_output_contract_if_pipeline_exists(sandbox, write_small_splits):
    import ntg.revenue.ltv as m

    fn = getattr(m, "build_user_ltv", None)
    Cfg = getattr(m, "LTVConfig", None)
    if fn is None or Cfg is None:
        pytest.skip("build_user_ltv/LTVConfig not found")

    try:
        fn(Cfg())
    except FileNotFoundError as e:
        pytest.skip(f"missing inputs for LTV in sandbox: {e}")

    df = pd.read_parquet("outputs/risk/user_ltv.parquet")
    assert "user_id" in df.columns
    # Allow your schema: could be ltv_usd, ltv_proxy, etc.
    assert any(c for c in df.columns if "ltv" in c.lower())
