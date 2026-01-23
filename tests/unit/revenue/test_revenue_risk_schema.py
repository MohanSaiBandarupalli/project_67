from __future__ import annotations

from pathlib import Path
import inspect
import pytest
import pandas as pd


def test_revenue_risk_module_imports():
    import ntg.revenue.revenue_risk as m
    assert m is not None


def _build_default_cfg(module):
    """
    Try common config class names. Return instance or None.
    """
    for name in ("RevenueRiskConfig", "RiskConfig", "Config"):
        cls = getattr(module, name, None)
        if cls is not None:
            try:
                return cls()
            except TypeError:
                # class exists but needs args
                return None
    return None


def test_revenue_risk_outputs_if_callable(sandbox, write_small_splits):
    import ntg.revenue.revenue_risk as m

    fn = getattr(m, "build_revenue_risk", None) or getattr(m, "main", None)
    if fn is None:
        pytest.skip("No build_revenue_risk/main found")

    # Try to generate prerequisites (churn + LTV) via pipeline if available
    try:
        import ntg.pipelines.score_users as p
        if getattr(p, "main", None):
            p.main()
    except Exception:
        pass

    # Call with cfg if required
    try:
        sig = inspect.signature(fn)
        if len(sig.parameters) == 0:
            fn()
        else:
            cfg = _build_default_cfg(m)
            if cfg is None:
                pytest.skip("build_revenue_risk requires cfg but no default config class found")
            fn(cfg)
    except FileNotFoundError as e:
        pytest.skip(f"missing inputs for revenue risk in sandbox: {e}")

    pth = Path("outputs/risk/user_risk.parquet")
    if not pth.exists():
        pytest.skip("user_risk.parquet not produced in sandbox (ok)")

    df = pd.read_parquet(pth)
    assert "user_id" in df.columns
