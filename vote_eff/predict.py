"""Predict majority-vote accuracy from an estimated correlation.

Prediction uses the normal approximation evaluated at the effective sample
size: the mean vote share is p, its variance is p * (1 - p) / n_eff, and
the majority is right when the share clears 0.5. The naive baseline is the
same formula at rho = 0 (n_eff = n) -- it is what you promise yourself when
you assume the samples are independent.

A consequence worth stating plainly: as n grows, n_eff tends to 1 / rho,
so under any positive correlation the vote accuracy has a ceiling strictly
below 1. Past a point, more samples buy nothing.
"""

import math
from statistics import NormalDist

from .estimators import effective_sample_size

_ND = NormalDist()


def _clip_p(p):
    return max(1e-9, min(1.0 - 1e-9, p))


def predicted_accuracy(p, rho, n):
    """Predicted majority-vote accuracy with n samples at correlation rho."""
    if n < 1:
        raise ValueError("n must be at least 1, got %r" % (n,))
    p = _clip_p(p)
    n_eff = effective_sample_size(n, rho)
    z = (p - 0.5) / math.sqrt(p * (1.0 - p) / n_eff)
    return _ND.cdf(z)


def predicted_gain(p, rho, n):
    """Predicted accuracy gain of an n-vote over a single sample."""
    return predicted_accuracy(p, rho, n) - predicted_accuracy(p, rho, 1)


def accuracy_ceiling(p, rho):
    """Best majority-vote accuracy reachable at any sample count."""
    rho = max(0.0, min(1.0, rho))
    if rho <= 0.0:
        return 1.0
    p = _clip_p(p)
    z = (p - 0.5) / math.sqrt(p * (1.0 - p) * rho)
    return _ND.cdf(z)


def samples_needed(p, rho, target, cap=1000000):
    """Smallest n whose predicted accuracy reaches target, or None.

    Returns None when the target sits above the correlation ceiling (no
    number of samples can reach it) or when p <= 0.5 (voting cannot help).
    """
    if p <= 0.5:
        return None
    if target <= predicted_accuracy(p, rho, 1):
        return 1
    if accuracy_ceiling(p, rho) < target:
        return None
    lo, hi = 1, 2
    while predicted_accuracy(p, rho, hi) < target:
        lo = hi
        hi *= 2
        if hi > cap:
            return None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if predicted_accuracy(p, rho, mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi
