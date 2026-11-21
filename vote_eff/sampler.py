"""Seeded vote sampler with a planted shared-bias component.

Each case draws one shared latent factor U ~ N(0, 1). Sample i of that case
is correct when

    sqrt(rho_latent) * U + sqrt(1 - rho_latent) * E_i  <  Phi^-1(p),

with E_i ~ N(0, 1) independent per sample. Because U is shared, the samples
of a case succeed or fail together in proportion to rho_latent: the shared
factor is the planted bias (a wrong attractor the whole population leans
toward). The correlation of the resulting correctness indicators is known
analytically through the bivariate normal CDF, which gives the audit an
exact ground truth to recover.
"""

import math
import random
from functools import lru_cache
from statistics import NormalDist

from .models import ProbeSet

_ND = NormalDist()
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    return _ND.cdf(x)


def norm_ppf(p):
    return _ND.inv_cdf(p)


def _phi(x):
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def bvn_cdf(a, b, rho):
    """P(X <= a, Y <= b) for a standard bivariate normal with correlation rho.

    Computed by Simpson integration of phi(x) * Phi((b - rho*x) / sqrt(1-rho^2))
    over x in [-10, a]. Deterministic; accurate to about 1e-7.
    """
    if rho == 0.0:
        return norm_cdf(a) * norm_cdf(b)
    rho = max(-0.999999, min(0.999999, rho))
    lo = -10.0
    if a <= lo:
        return 0.0
    s = math.sqrt(1.0 - rho * rho)

    def integrand(x):
        return _phi(x) * norm_cdf((b - rho * x) / s)

    intervals = 4000
    h = (a - lo) / intervals
    total = integrand(lo) + integrand(a)
    for i in range(1, intervals):
        coef = 4 if i % 2 == 1 else 2
        total += coef * integrand(lo + i * h)
    return total * h / 3.0


@lru_cache(maxsize=None)
def indicator_correlation(p, rho_latent):
    """Exact correlation of two correctness indicators from one case.

    For threshold t = Phi^-1(p), the joint probability that two samples are
    both correct is Phi_2(t, t; rho_latent), so the indicator correlation is
    (Phi_2(t, t; rho_latent) - p^2) / (p * (1 - p)).
    """
    if rho_latent == 0.0:
        return 0.0
    t = norm_ppf(p)
    joint = bvn_cdf(t, t, rho_latent)
    rho = (joint - p * p) / (p * (1.0 - p))
    return max(0.0, min(1.0, rho))


def analytic_effective_n(n, p, rho_latent):
    """Ground-truth effective sample size under the planted bias."""
    rho = indicator_correlation(p, rho_latent)
    return n / (1.0 + (n - 1) * rho)


class VoteSampler:
    """Draws votes from one population."""

    def __init__(self, population):
        self.population = population
        self._threshold = norm_ppf(population.p)
        self._shared = math.sqrt(population.rho_latent)
        self._idio = math.sqrt(1.0 - population.rho_latent)

    def draw_case(self, rng, n_samples):
        shared = self._shared * rng.gauss(0.0, 1.0)
        threshold = self._threshold
        idio = self._idio
        return [
            (shared + idio * rng.gauss(0.0, 1.0)) < threshold
            for _ in range(n_samples)
        ]


def generate_probe(population, n_cases, n_samples, seed):
    """Generate a labelled probe set from a population."""
    rng = random.Random(seed)
    sampler = VoteSampler(population)
    votes = [sampler.draw_case(rng, n_samples) for _ in range(n_cases)]
    return ProbeSet(votes=votes)


def simulate_accuracy_curve(population, targets, n_cases, seed):
    """Realised majority-vote accuracy at each target n, by simulation.

    Draws max(targets) votes per case once and scores every prefix target
    from the same draws. A tie at even n earns half credit.
    """
    targets = sorted(set(int(n) for n in targets))
    if not targets or targets[0] < 1:
        raise ValueError("targets must be positive sample counts")
    rng = random.Random(seed)
    sampler = VoteSampler(population)
    max_n = targets[-1]
    target_set = set(targets)
    totals = dict((n, 0.0) for n in targets)
    for _ in range(n_cases):
        votes = sampler.draw_case(rng, max_n)
        running = 0
        for i, value in enumerate(votes, start=1):
            if value:
                running += 1
            if i in target_set:
                if 2 * running > i:
                    totals[i] += 1.0
                elif 2 * running == i:
                    totals[i] += 0.5
    return dict((n, totals[n] / n_cases) for n in targets)
