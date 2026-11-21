import pytest

from vote_eff.estimators import estimate_accuracy, estimate_rho_agreement
from vote_eff.models import Population
from vote_eff.sampler import (
    analytic_effective_n,
    bvn_cdf,
    generate_probe,
    indicator_correlation,
    norm_cdf,
    simulate_accuracy_curve,
)


def test_bvn_cdf_at_zero_rho_is_product():
    assert bvn_cdf(0.3, -0.2, 0.0) == pytest.approx(norm_cdf(0.3) * norm_cdf(-0.2))


def test_bvn_cdf_origin():
    assert bvn_cdf(0.0, 0.0, 0.0) == pytest.approx(0.25)


def test_bvn_cdf_symmetric():
    assert bvn_cdf(0.4, -0.7, 0.5) == pytest.approx(bvn_cdf(-0.7, 0.4, 0.5), abs=1e-6)


def test_bvn_cdf_bounded_by_marginals():
    value = bvn_cdf(0.2, 0.9, 0.8)
    assert 0.0 <= value <= min(norm_cdf(0.2), norm_cdf(0.9))


def test_indicator_correlation_zero_without_bias():
    assert indicator_correlation(0.7, 0.0) == 0.0


def test_indicator_correlation_increases_with_bias():
    values = [indicator_correlation(0.7, rho) for rho in (0.1, 0.3, 0.6, 0.9)]
    assert values == sorted(values)
    assert values[0] > 0.0
    assert values[-1] > 0.6


def test_indicator_correlation_near_one_at_max_bias():
    assert indicator_correlation(0.7, 0.999) > 0.95


def test_analytic_effective_n():
    assert analytic_effective_n(32, 0.7, 0.0) == pytest.approx(32.0)
    assert analytic_effective_n(32, 0.7, 0.9) < 2.0
    assert analytic_effective_n(1, 0.7, 0.9) == pytest.approx(1.0)


def test_generate_probe_is_deterministic():
    pop = Population(p=0.7, rho_latent=0.6)
    first = generate_probe(pop, 12, 8, 42)
    second = generate_probe(pop, 12, 8, 42)
    assert first.votes == second.votes


def test_generate_probe_seed_changes_output():
    pop = Population(p=0.7, rho_latent=0.6)
    first = generate_probe(pop, 12, 8, 42)
    second = generate_probe(pop, 12, 8, 43)
    assert first.votes != second.votes


def test_empirical_accuracy_matches_population():
    pop = Population(p=0.7, rho_latent=0.0)
    probe = generate_probe(pop, 400, 25, 11)
    assert estimate_accuracy(probe) == pytest.approx(0.7, abs=0.02)


def test_empirical_rho_recovery():
    pop = Population(p=0.7, rho_latent=0.6)
    probe = generate_probe(pop, 600, 16, 13)
    rho_hat = estimate_rho_agreement(probe)
    assert rho_hat == pytest.approx(indicator_correlation(0.7, 0.6), abs=0.08)


def test_simulate_accuracy_curve_independent_votes_help():
    pop = Population(p=0.7, rho_latent=0.0)
    curve = simulate_accuracy_curve(pop, [1, 32], 4000, 17)
    assert curve[1] == pytest.approx(0.7, abs=0.03)
    assert curve[32] > curve[1] + 0.15


def test_simulate_accuracy_curve_biased_votes_barely_help():
    pop = Population(p=0.7, rho_latent=0.9)
    curve = simulate_accuracy_curve(pop, [1, 32], 4000, 19)
    assert curve[32] - curve[1] < 0.08
