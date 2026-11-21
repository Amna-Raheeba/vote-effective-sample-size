"""Command line interface.

Usage:
    python -m vote_eff demo
    python -m vote_eff audit --probe examples/demo_probe.json
    python -m vote_eff evaluate --outdir results
"""

import argparse
import json

from .audit import run_audit
from .evaluate import run_evaluation, write_results
from .models import Population, ProbeSet
from .report import audit_report_markdown
from .sampler import generate_probe


def build_parser():
    parser = argparse.ArgumentParser(
        prog="vote_eff",
        description=(
            "Audit self-consistency voting: convert agreement into an "
            "effective number of independent votes and predict vote gain."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Audit a built-in demo population (p=0.70, bias 0.6)")

    audit_parser = sub.add_parser("audit", help="Audit a labelled probe JSON file")
    audit_parser.add_argument("--probe", required=True, help="Path to probe JSON")
    audit_parser.add_argument(
        "--targets",
        default="8,16,32,64",
        help="Comma-separated target sample counts (default 8,16,32,64)",
    )
    audit_parser.add_argument(
        "--min-gain",
        type=float,
        default=0.02,
        help="Minimum predicted gain worth paying for (default 0.02)",
    )

    eval_parser = sub.add_parser(
        "evaluate", help="Run the headline evaluation and write results"
    )
    eval_parser.add_argument("--outdir", default="results")
    return parser


def _parse_targets(text):
    return tuple(int(part) for part in text.split(",") if part.strip())


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        population = Population(p=0.70, rho_latent=0.6, seed=7)
        probe = generate_probe(population, 48, 16, 8)
        result = run_audit(probe)
        print(audit_report_markdown(result))
        return 0
    if args.command == "audit":
        with open(args.probe) as handle:
            probe = ProbeSet.from_dict(json.load(handle))
        result = run_audit(
            probe, targets=_parse_targets(args.targets), min_gain=args.min_gain
        )
        print(audit_report_markdown(result))
        return 0
    if args.command == "evaluate":
        evaluation = run_evaluation()
        write_results(evaluation, args.outdir)
        agg = evaluation["aggregates"]
        print(
            "Wrote results to %s (gain MAE tool %.4f vs naive %.4f)"
            % (args.outdir, agg["gain_mae_tool"], agg["gain_mae_naive"])
        )
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
