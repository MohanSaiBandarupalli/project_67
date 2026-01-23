from __future__ import annotations

import math


def test_mde_required_n_basic_sanity():
    """
    Simple sanity: required n increases when std increases (all else equal).
    """
    # approximate normal z for alpha=0.05, power=0.8
    z_alpha = 1.96
    z_beta = 0.84

    def required_n(std: float, mde: float) -> float:
        return 2 * ((z_alpha + z_beta) * std / mde) ** 2

    n1 = required_n(std=1.0, mde=0.1)
    n2 = required_n(std=2.0, mde=0.1)
    assert n2 > n1
