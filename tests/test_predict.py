import pytest

from vote_eff.predict import (
    accuracy_ceiling,
    predicted_accuracy,
    predicted_gain,
    samples_needed,
)


def test_predicted_accuracy_increases_with_n_when_independent():
    values = [predicted_accuracy(0.7, 0.0, n) for n in (1, 4, 16, 64)]
    assert values == sorted(values)
    assert values[-1] > 0.95


def test_predicted_accuracy_barely_increases_when_correlated():
    gain = predicted_accuracy(0.7, 0.9, 64) - predicted_accuracy(0.7, 0.9, 1)
    assert gain < 0.05


def test_predicted_gain_at_one_is_zero():
    assert predicted_gain(0.7, 0.4, 1) == pytest.approx(0.0)


def test_naive_gain_exceeds_correlated_gain():
    assert predicted_gain(0.7, 0.0, 32) > predicted_gain(0.7, 0.6, 32)


def test_predicted_accuracy_rejects_zero_n():
    with pytest.raises(ValueError):
        predicted_accuracy(0.7, 0.3, 0)


def test_ceiling_is_one_without_correlation():
    assert accuracy_ceiling(0.7, 0.0) == 1.0


def test_ceiling_below_one_with_correlation():
    ceiling = accuracy_ceiling(0.7, 0.4)
    assert 0.7 < ceiling < 1.0


def test_ceiling_bounds_all_predictions():
    ceiling = accuracy_ceiling(0.8, 0.5)
    for n in (1, 8, 64, 512):
        assert predicted_accuracy(0.8, 0.5, n) <= ceiling + 1e-9


def test_samples_needed_reachable():
    n = samples_needed(0.7, 0.0, 0.99)
    assert n is not None
    assert predicted_accuracy(0.7, 0.0, n) >= 0.99
    assert predicted_accuracy(0.7, 0.0, n - 1) < 0.99


def test_samples_needed_unreachable_above_ceiling():
    assert samples_needed(0.7, 0.7, 0.99) is None


def test_samples_needed_none_at_or_below_chance():
    assert samples_needed(0.5, 0.0, 0.9) is None
    assert samples_needed(0.4, 0.0, 0.9) is None


def test_samples_needed_already_there():
    assert samples_needed(0.9, 0.3, 0.80) == 1
