from __future__ import annotations


def test_roi_gate_behavior_sanity():
    """
    If net value is negative, ROI gate should reject.
    """
    def roi_gate(mean_net_value: float) -> bool:
        return mean_net_value > 0.0

    assert roi_gate(0.1) is True
    assert roi_gate(-0.01) is False
