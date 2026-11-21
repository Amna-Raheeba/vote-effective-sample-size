"""Reproduce the headline results and check them for drift.

Re-runs the full evaluation with the canonical seeds and compares the
aggregate numbers against results/evaluation.json. Exits 0 when every
drift key is within tolerance, 1 otherwise.

Run: python scripts/reproduce_results.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vote_eff.evaluate import check_drift, run_evaluation


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    stored_path = os.path.join(root, "results", "evaluation.json")
    with open(stored_path) as handle:
        stored = json.load(handle)
    fresh = run_evaluation()
    ok, problems = check_drift(stored, fresh)
    if ok:
        print("PASS: headline aggregates reproduced within tolerance.")
        return 0
    print("FAIL: headline drift detected:")
    for key, stored_value, fresh_value, diff in problems:
        print(
            "  %s: stored=%r fresh=%r diff=%r" % (key, stored_value, fresh_value, diff)
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
