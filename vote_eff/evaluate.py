"""Headline evaluation: can the audit recover n_eff and predict vote gain?

Each scenario plants a known accuracy p and latent bias rho_latent. The
audit sees only a small labelled probe (48 cases x 16 samples). Its
estimates are scored against (a) the analytic ground truth for the
indicator correlation and n_eff, and (b) the realised majority-vote
accuracy from a large seeded simulation. The naive baseline predicts with
rho = 0, i.e. it assumes every sample is an independent vote.
"""

import csv
import json
import os

from .audit import run_audit
from .estimators import effective_sample_size
from .models import Population
from .sampler import (
    analytic_effective_n,
    generate_probe,
    indicator_correlation,
    simulate_accuracy_curve,
)

DEFAULT_CONFIG = {
    "p_values": [0.60, 0.70, 0.80],
    "rho_values": [0.0, 0.3, 0.6, 0.9],
    "probe_cases": 48,
    "probe_samples": 16,
    "realised_cases": 3000,
    "targets": [16, 32, 64],
    "base_seed": 7,
}

DRIFT_KEYS = [
    "gain_mae_tool",
    "gain_mae_naive",
    "mean_neff_rel_error_rho_positive",
    "naive_overpromise_high_rho",
]


def run_evaluation(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    targets = [int(n) for n in cfg["targets"]]
    scenarios = []
    index = 0
    for p in cfg["p_values"]:
        for rho_latent in cfg["rho_values"]:
            seed = cfg["base_seed"] + 1000 * index
            population = Population(p=p, rho_latent=rho_latent, seed=seed)
            probe = generate_probe(
                population, cfg["probe_cases"], cfg["probe_samples"], seed + 1
            )
            audit = run_audit(probe, targets=tuple(targets))
            curve = simulate_accuracy_curve(
                population, [1] + targets, cfg["realised_cases"], seed + 2
            )
            rho_true = indicator_correlation(p, rho_latent)
            neff_true_32 = analytic_effective_n(32, p, rho_latent)
            neff_hat_32 = effective_sample_size(32, audit.rho_hat)
            if neff_true_32 > 0:
                neff_rel_error = abs(neff_hat_32 - neff_true_32) / neff_true_32
            else:
                neff_rel_error = 0.0
            target_rows = []
            for n in targets:
                row = audit.row_for(n)
                realised_gain = curve[n] - curve[1]
                target_rows.append(
                    {
                        "n": n,
                        "n_eff_hat": row.n_eff,
                        "realised_accuracy": curve[n],
                        "realised_gain": realised_gain,
                        "tool_pred_accuracy": row.pred_accuracy,
                        "tool_pred_gain": row.pred_gain,
                        "tool_gain_error": row.pred_gain - realised_gain,
                        "naive_pred_accuracy": row.naive_accuracy,
                        "naive_pred_gain": row.naive_gain,
                        "naive_gain_error": row.naive_gain - realised_gain,
                    }
                )
            scenarios.append(
                {
                    "p": p,
                    "rho_latent": rho_latent,
                    "rho_true": rho_true,
                    "p_hat": audit.p_hat,
                    "rho_hat": audit.rho_hat,
                    "rho_hat_variance": audit.rho_hat_variance,
                    "bias_share": audit.bias_share,
                    "ceiling_hat": audit.ceiling,
                    "neff_true_32": neff_true_32,
                    "neff_hat_32": neff_hat_32,
                    "neff_rel_error_32": neff_rel_error,
                    "single_accuracy_realised": curve[1],
                    "target_rows": target_rows,
                }
            )
            index += 1
    return {
        "config": cfg,
        "scenarios": scenarios,
        "aggregates": _aggregates(scenarios),
    }


def _aggregates(scenarios):
    tool_errors = []
    naive_errors = []
    tool_signed = []
    naive_signed = []
    naive_over_high = []
    neff_errors = []
    for scenario in scenarios:
        if scenario["rho_latent"] > 0.0:
            neff_errors.append(scenario["neff_rel_error_32"])
        for row in scenario["target_rows"]:
            tool_errors.append(abs(row["tool_gain_error"]))
            naive_errors.append(abs(row["naive_gain_error"]))
            tool_signed.append(row["tool_gain_error"])
            naive_signed.append(row["naive_gain_error"])
            if scenario["rho_latent"] >= 0.6:
                naive_over_high.append(row["naive_gain_error"])
    showcase = None
    for scenario in scenarios:
        if scenario["p"] == 0.70 and scenario["rho_latent"] == 0.9:
            row32 = None
            for row in scenario["target_rows"]:
                if row["n"] == 32:
                    row32 = row
            showcase = {
                "p": scenario["p"],
                "rho_latent": scenario["rho_latent"],
                "rho_true": scenario["rho_true"],
                "rho_hat": scenario["rho_hat"],
                "neff_true_32": scenario["neff_true_32"],
                "neff_hat_32": scenario["neff_hat_32"],
                "realised_gain_32": row32["realised_gain"] if row32 else None,
                "tool_pred_gain_32": row32["tool_pred_gain"] if row32 else None,
                "naive_pred_gain_32": row32["naive_pred_gain"] if row32 else None,
            }

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "n_scenarios": len(scenarios),
        "gain_mae_tool": mean(tool_errors),
        "gain_mae_naive": mean(naive_errors),
        "gain_bias_tool": mean(tool_signed),
        "gain_bias_naive": mean(naive_signed),
        "naive_overpromise_high_rho": mean(naive_over_high),
        "mean_neff_rel_error_rho_positive": mean(neff_errors),
        "showcase": showcase,
    }


def write_results(evaluation, outdir):
    """Write evaluation.json / .csv, evaluation_summary.md and HEADLINE.md."""
    os.makedirs(outdir, exist_ok=True)
    from .report import evaluation_summary_markdown, headline_markdown

    with open(os.path.join(outdir, "evaluation.json"), "w") as handle:
        json.dump(evaluation, handle, indent=2, sort_keys=True)
    with open(os.path.join(outdir, "evaluation.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "rho_latent",
                "rho_true",
                "rho_hat",
                "neff_true_32",
                "neff_hat_32",
                "n",
                "realised_gain",
                "tool_pred_gain",
                "tool_gain_error",
                "naive_pred_gain",
                "naive_gain_error",
            ]
        )
        for scenario in evaluation["scenarios"]:
            for row in scenario["target_rows"]:
                writer.writerow(
                    [
                        scenario["p"],
                        scenario["rho_latent"],
                        "%.6f" % scenario["rho_true"],
                        "%.6f" % scenario["rho_hat"],
                        "%.4f" % scenario["neff_true_32"],
                        "%.4f" % scenario["neff_hat_32"],
                        row["n"],
                        "%.6f" % row["realised_gain"],
                        "%.6f" % row["tool_pred_gain"],
                        "%.6f" % row["tool_gain_error"],
                        "%.6f" % row["naive_pred_gain"],
                        "%.6f" % row["naive_gain_error"],
                    ]
                )
    with open(os.path.join(outdir, "evaluation_summary.md"), "w") as handle:
        handle.write(evaluation_summary_markdown(evaluation))
    with open(os.path.join(outdir, "HEADLINE.md"), "w") as handle:
        handle.write(headline_markdown(evaluation))
    return outdir


def check_drift(stored, fresh, tol=0.02):
    """Compare stored vs freshly computed aggregates.

    Returns (ok, problems) where problems lists (key, stored, fresh, diff)
    for every drift key whose absolute difference exceeds tol.
    """
    problems = []
    stored_agg = stored.get("aggregates", {})
    fresh_agg = fresh.get("aggregates", {})
    for key in DRIFT_KEYS:
        if key not in stored_agg or key not in fresh_agg:
            problems.append((key, stored_agg.get(key), fresh_agg.get(key), None))
            continue
        diff = abs(stored_agg[key] - fresh_agg[key])
        if diff > tol:
            problems.append((key, stored_agg[key], fresh_agg[key], diff))
    return (len(problems) == 0, problems)
