"""Demo: audit one population and print the full report.

The demo population answers correctly 70% of the time, but its samples
share a strong planted bias (latent correlation 0.6), so a 32-sample vote
is worth far fewer than 32 independent votes.

Run: python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vote_eff.audit import run_audit
from vote_eff.models import Population
from vote_eff.predict import accuracy_ceiling, samples_needed
from vote_eff.report import audit_report_markdown
from vote_eff.sampler import generate_probe


def main():
    population = Population(p=0.70, rho_latent=0.6, seed=7)
    probe = generate_probe(population, 48, 16, 8)
    result = run_audit(probe)
    print(audit_report_markdown(result))
    print(
        "Accuracy ceiling at the estimated correlation: %.3f"
        % accuracy_ceiling(result.p_hat, result.rho_hat)
    )
    needed = samples_needed(result.p_hat, result.rho_hat, 0.90)
    if needed is None:
        print("Reaching 0.90 accuracy by voting alone: impossible at this correlation.")
    else:
        print("Samples needed to reach 0.90 predicted accuracy: %d" % needed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
