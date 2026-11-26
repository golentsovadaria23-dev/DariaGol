"""Small statistical helpers used in the pilot-analysis notebooks."""

import math
from statistics import NormalDist


z = NormalDist().inv_cdf


def n_per_group(
    sigma: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> int:
    """Return the required sample size for each of two equally sized groups."""
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma**2 / mde**2)


def mde_for_n(
    sigma: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> float:
    """Return the minimum detectable difference for ``n`` clients per group."""
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return (2 / n) ** 0.5 * (za + zb) * sigma


def diff_ci(t, c, alpha: float = 0.05) -> dict:
    """Return the test-control mean difference, normal CI, and two-sided p-value."""
    d = t.mean() - c.mean()
    se = (t.var() / len(t) + c.var() / len(c)) ** 0.5
    q = z(1 - alpha / 2)
    p = 2 * (1 - NormalDist().cdf(abs(d / se)))
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "p": p}


def srm_pvalue(sizes) -> float:
    """Return the equal-allocation chi-square p-value for two or three groups."""
    n = sum(sizes)
    expected = n / len(sizes)
    chi2 = sum((size - expected) ** 2 / expected for size in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    if len(sizes) == 3:
        return math.exp(-chi2 / 2)
    raise ValueError("для 4 и более групп — scipy.stats.chisquare")


def cuped(y, x):
    """Return the CUPED-adjusted metric and its fitted theta coefficient."""
    theta = y.cov(x) / x.var()
    return y - theta * (x - x.mean()), theta
