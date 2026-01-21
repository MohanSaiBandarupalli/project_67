from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ExperimentDesign:
    """
    Simple, defensible experiment design for retention / revenue metrics.

    primary metric: revenue_at_risk_usd (lower is better)
    secondary: churn_prob (p_churn) (lower is better)

    We model revenue_at_risk as a continuous metric and use
    a normal approximation for sample size + z-test.
    """
    alpha: float = 0.05         # type-I error (two-sided)
    power: float = 0.80         # 1 - beta
    mde_pct: float = 0.03       # minimum detectable effect as % lift (e.g., 3% reduction)
    two_sided: bool = True

    # If you want to constrain how long the experiment runs, you can derive n/day later.


def _z_from_alpha(alpha: float, two_sided: bool) -> float:
    # Inverse normal CDF approx via numpy
    # We'll use scipy-less approximation by sampling a large grid isn't great.
    # Instead, use an accurate rational approximation.
    # Source: Peter John Acklam’s approximation (implemented inline).
    p = 1 - (alpha / 2 if two_sided else alpha)
    return float(_norminv(p))


def _z_from_power(power: float) -> float:
    return float(_norminv(power))


def _norminv(p: float) -> float:
    """
    Approximate inverse CDF of standard normal distribution.
    Acklam's approximation; accurate enough for experiment design.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")

    # Coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def required_sample_size_continuous(
    baseline_mean: float,
    baseline_std: float,
    design: ExperimentDesign,
) -> int:
    """
    Sample size per group for detecting relative change (mde_pct) in a continuous metric.

    n ≈ 2 * ( (z_alpha + z_beta) * sigma / delta )^2
    where delta = baseline_mean * mde_pct
    """
    if baseline_mean <= 0:
        raise ValueError("baseline_mean must be > 0")
    if baseline_std <= 0:
        raise ValueError("baseline_std must be > 0")

    z_alpha = _z_from_alpha(design.alpha, design.two_sided)
    z_beta = _z_from_power(design.power)
    delta = baseline_mean * design.mde_pct

    n = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2
    return int(math.ceil(n))


def ztest_diff_means(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Two-sample z-test with normal approximation (good when n is large).
    Returns effect (y-x), z, p_value, and percent lift relative to x mean.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    nx, ny = len(x), len(y)

    se = math.sqrt(vx / nx + vy / ny)
    if se == 0:
        return {"mx": mx, "my": my, "effect": my - mx, "z": float("nan"), "p_value": float("nan"), "lift_pct": float("nan")}

    z = (my - mx) / se

    # p-value using normal CDF
    p = 2 * (1 - _normcdf(abs(z)))

    lift = (my - mx) / mx if mx != 0 else float("nan")

    return {"mx": mx, "my": my, "effect": my - mx, "z": float(z), "p_value": float(p), "lift_pct": float(lift)}


def _normcdf(x: float) -> float:
    # standard normal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
