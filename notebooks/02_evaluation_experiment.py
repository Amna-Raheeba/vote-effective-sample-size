"""Evaluation experiment (reduced smoke version).

Runs the headline evaluation with fewer realised cases so it finishes in
seconds, and prints the aggregate numbers. The full run that produced
results/ is `python -m vote_eff evaluate`.

Run: python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vote_eff.evaluate import run_evaluation


def main():
    evaluation = run_evaluation({"realised_cases": 800})
    agg = evaluation["aggregates"]
    print("Scenarios: %d" % agg["n_scenarios"])
    print("Gain prediction MAE, tool:  %.4f" % agg["gain_mae_tool"])
    print("Gain prediction MAE, naive: %.4f" % agg["gain_mae_naive"])
    print("Naive over-promise at high bias: %+.4f" % agg["naive_overpromise_high_rho"])
    print(
        "Mean n_eff relative error at n=32 (biased): %.1f%%"
        % (100.0 * agg["mean_neff_rel_error_rho_positive"])
    )
    showcase = agg["showcase"]
    if showcase:
        print(
            "Showcase p=%.2f bias=%.1f: 32 samples worth %.2f independent "
            "votes (audit estimate %.2f)"
            % (
                showcase["p"],
                showcase["rho_latent"],
                showcase["neff_true_32"],
                showcase["neff_hat_32"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
