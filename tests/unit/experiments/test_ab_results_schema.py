from __future__ import annotations

import json
import pytest


def test_ab_results_schema_if_present(project_root):
    p = project_root / "reports/ab_results_3arm.json"
    if not p.exists():
        pytest.skip("ab_results_3arm.json not present in repo")
    obj = json.loads(p.read_text(encoding="utf-8"))

    assert "experiment_id" in obj
    assert "counts" in obj
    assert "design" in obj
    assert "tests_vs_control" in obj
