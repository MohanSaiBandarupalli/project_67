from __future__ import annotations

import pytest


def test_experiments_module_imports():
    # if experiments package exists
    import ntg.experiments
    assert ntg.experiments is not None


def test_3arm_pipeline_imports():
    import ntg.pipelines.run_experiment_3arm as m
    assert m is not None


def test_3arm_bucketing_is_deterministic(sandbox, write_small_splits):
    import json
    import os

    import ntg.pipelines.run_experiment_3arm as m

    run = getattr(m, "main", None)
    if run is None:
        pytest.skip("run_experiment_3arm.main not found")

    # requires user_risk + interventions; if missing, skip
    try:
        run()
    except FileNotFoundError as e:
        pytest.skip(f"missing prerequisites for 3-arm experiment: {e}")

    a = json.loads(open("reports/ab_results_3arm.json", "r", encoding="utf-8").read())

    # Run again (overwrite outputs)
    run()
    b = json.loads(open("reports/ab_results_3arm.json", "r", encoding="utf-8").read())

    assert a["counts"] == b["counts"]
    assert a["experiment_id"] == b["experiment_id"]
