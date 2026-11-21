import json
import os

from vote_eff.evaluate import run_evaluation, write_results

SMALL_CONFIG = {
    "p_values": [0.70],
    "rho_values": [0.0, 0.9],
    "probe_cases": 24,
    "probe_samples": 12,
    "realised_cases": 400,
    "targets": [16, 32],
    "base_seed": 7,
}


def test_run_evaluation_shape():
    evaluation = run_evaluation(SMALL_CONFIG)
    assert len(evaluation["scenarios"]) == 2
    agg = evaluation["aggregates"]
    assert agg["n_scenarios"] == 2
    for key in (
        "gain_mae_tool",
        "gain_mae_naive",
        "gain_bias_tool",
        "gain_bias_naive",
        "naive_overpromise_high_rho",
        "mean_neff_rel_error_rho_positive",
    ):
        assert key in agg
    for scenario in evaluation["scenarios"]:
        assert len(scenario["target_rows"]) == 2
        assert 0.0 <= scenario["rho_hat"] <= 1.0
        assert scenario["neff_hat_32"] >= 1.0


def test_run_evaluation_showcase_present():
    evaluation = run_evaluation(SMALL_CONFIG)
    showcase = evaluation["aggregates"]["showcase"]
    assert showcase is not None
    assert showcase["p"] == 0.70
    assert showcase["rho_latent"] == 0.9
    assert showcase["neff_true_32"] < 2.0


def test_run_evaluation_naive_overpromises_at_high_bias():
    evaluation = run_evaluation(SMALL_CONFIG)
    biased = [s for s in evaluation["scenarios"] if s["rho_latent"] == 0.9][0]
    for row in biased["target_rows"]:
        assert row["naive_pred_gain"] > row["tool_pred_gain"]


def test_write_results_creates_all_files(tmp_path):
    evaluation = run_evaluation(SMALL_CONFIG)
    outdir = str(tmp_path / "results")
    write_results(evaluation, outdir)
    for name in (
        "evaluation.json",
        "evaluation.csv",
        "evaluation_summary.md",
        "HEADLINE.md",
    ):
        assert os.path.exists(os.path.join(outdir, name))
    with open(os.path.join(outdir, "evaluation.json")) as handle:
        stored = json.load(handle)
    assert stored["aggregates"]["n_scenarios"] == 2
