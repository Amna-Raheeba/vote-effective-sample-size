import copy

from vote_eff.evaluate import DRIFT_KEYS, check_drift, run_evaluation

SMALL_CONFIG = {
    "p_values": [0.70],
    "rho_values": [0.0, 0.9],
    "probe_cases": 24,
    "probe_samples": 12,
    "realised_cases": 400,
    "targets": [16, 32],
    "base_seed": 7,
}


def test_check_drift_identical_passes():
    evaluation = run_evaluation(SMALL_CONFIG)
    ok, problems = check_drift(evaluation, evaluation)
    assert ok
    assert problems == []


def test_check_drift_detects_perturbation():
    evaluation = run_evaluation(SMALL_CONFIG)
    perturbed = copy.deepcopy(evaluation)
    perturbed["aggregates"]["gain_mae_naive"] += 0.5
    ok, problems = check_drift(evaluation, perturbed)
    assert not ok
    keys = [problem[0] for problem in problems]
    assert "gain_mae_naive" in keys


def test_check_drift_missing_key_fails():
    evaluation = run_evaluation(SMALL_CONFIG)
    broken = copy.deepcopy(evaluation)
    del broken["aggregates"][DRIFT_KEYS[0]]
    ok, problems = check_drift(evaluation, broken)
    assert not ok
