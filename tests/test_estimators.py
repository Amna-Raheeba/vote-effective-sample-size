import pytest

from vote_eff.estimators import (
    agreement_decomposition,
    effective_sample_size,
    estimate_accuracy,
    estimate_rho_agreement,
    estimate_rho_variance,
    independent_agreement,
    pairwise_agreement,
)
from vote_eff.models import ProbeSet


def _probe_from_counts(counts, n_samples):
    votes = []
    for k in counts:
        votes.append([True] * k + [False] * (n_samples - k))
    return ProbeSet(votes=votes)


def test_estimate_accuracy():
    probe = ProbeSet(votes=[[True, False], [True, True]])
    assert estimate_accuracy(probe) == pytest.approx(0.75)


def test_pairwise_agreement_unanimous():
    probe = ProbeSet(votes=[[True, True, True], [False, False, False]])
    assert pairwise_agreement(probe) == pytest.approx(1.0)


def test_pairwise_agreement_split():
    probe = ProbeSet(votes=[[True, False], [True, False]])
    assert pairwise_agreement(probe) == pytest.approx(0.0)


def test_independent_agreement_values():
    assert independent_agreement(0.5) == pytest.approx(0.5)
    assert independent_agreement(0.7) == pytest.approx(0.58)
    assert independent_agreement(1.0) == pytest.approx(1.0)


def test_rho_agreement_exact_construction():
    # counts [4, 0, 3, 1] of 4: p_hat = 0.5, agreement = 0.75 exactly,
    # so rho = (0.75 - 0.5) / (2 * 0.25) = 0.5 exactly.
    probe = _probe_from_counts([4, 0, 3, 1], 4)
    assert estimate_accuracy(probe) == pytest.approx(0.5)
    assert pairwise_agreement(probe) == pytest.approx(0.75)
    assert estimate_rho_agreement(probe) == pytest.approx(0.5)


def test_rho_agreement_unanimous_cases_give_one():
    probe = _probe_from_counts([4, 0, 4, 0], 4)
    assert estimate_rho_agreement(probe) == pytest.approx(1.0)


def test_rho_variance_exact_construction():
    # counts [4, 0, 3, 1]: sample variance 10/3, so
    # rho = ((10/3) / (4 * 0.25) - 1) / 3 = 7/9.
    probe = _probe_from_counts([4, 0, 3, 1], 4)
    assert estimate_rho_variance(probe) == pytest.approx(7.0 / 9.0)


def test_rho_estimators_degenerate_probe():
    probe = ProbeSet(votes=[[True, True], [True, True]])
    assert estimate_rho_agreement(probe) == 0.0
    assert estimate_rho_variance(probe) == 0.0


def test_rho_agreement_clipped_to_unit_interval():
    probe = _probe_from_counts([4, 0, 4, 0], 4)
    rho = estimate_rho_agreement(probe)
    assert 0.0 <= rho <= 1.0


def test_effective_sample_size_formula():
    assert effective_sample_size(32, 0.0) == pytest.approx(32.0)
    assert effective_sample_size(32, 1.0) == pytest.approx(1.0)
    assert effective_sample_size(32, 0.5) == pytest.approx(32.0 / 16.5)
    assert effective_sample_size(1, 0.9) == pytest.approx(1.0)


def test_effective_sample_size_rejects_zero():
    with pytest.raises(ValueError):
        effective_sample_size(0, 0.5)


def test_agreement_decomposition_sums():
    probe = _probe_from_counts([4, 0, 3, 1], 4)
    decomp = agreement_decomposition(probe)
    assert decomp["agreement"] == pytest.approx(
        decomp["independent_agreement"] + decomp["excess_agreement"]
    )
    assert decomp["bias_share"] == pytest.approx(0.25 / 0.75)
    assert decomp["p_hat"] == pytest.approx(0.5)
