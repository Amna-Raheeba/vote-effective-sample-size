"""Run the audit on the bundled demo probe and print the report.

Run: python examples/run_audit.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vote_eff.audit import run_audit
from vote_eff.models import ProbeSet
from vote_eff.report import audit_report_markdown


def main():
    path = os.path.join(os.path.dirname(__file__), "demo_probe.json")
    with open(path) as handle:
        probe = ProbeSet.from_dict(json.load(handle))
    result = run_audit(probe)
    print(audit_report_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
