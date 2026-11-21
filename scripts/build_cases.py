"""Generate the stored probe data.

Writes:
- data/probe_sets.json: the probe set for every evaluation scenario, using
  the same seed scheme as vote_eff.evaluate, so the stored probes are
  exactly the ones the headline evaluation audits.
- examples/demo_probe.json: one demo probe (p = 0.70, latent bias 0.6).

Run: python scripts/build_cases.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vote_eff.evaluate import DEFAULT_CONFIG
from vote_eff.models import Population
from vote_eff.sampler import generate_probe


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = DEFAULT_CONFIG
    scenarios = []
    index = 0
    for p in cfg["p_values"]:
        for rho_latent in cfg["rho_values"]:
            seed = cfg["base_seed"] + 1000 * index
            population = Population(p=p, rho_latent=rho_latent, seed=seed)
            probe = generate_probe(
                population, cfg["probe_cases"], cfg["probe_samples"], seed + 1
            )
            scenarios.append(
                {
                    "p": p,
                    "rho_latent": rho_latent,
                    "probe": probe.to_dict(),
                }
            )
            index += 1
    data_path = os.path.join(root, "data", "probe_sets.json")
    with open(data_path, "w") as handle:
        json.dump({"scenarios": scenarios}, handle)
    print("Wrote %s (%d scenarios)" % (data_path, len(scenarios)))

    demo_population = Population(p=0.70, rho_latent=0.6, seed=7)
    demo_probe = generate_probe(demo_population, 48, 16, 8)
    demo_path = os.path.join(root, "examples", "demo_probe.json")
    with open(demo_path, "w") as handle:
        json.dump(demo_probe.to_dict(), handle)
    print("Wrote %s" % demo_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
