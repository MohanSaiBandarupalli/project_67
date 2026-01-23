from __future__ import annotations

import json
from pathlib import Path


def test_ranking_metrics_json_is_valid_if_exists():
    p = Path("reports/ranking_metrics.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(obj, dict)
