import json
import os

from vote_eff.cli import main


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Self-consistency vote audit" in out
    assert "Recommendation" in out


def test_cli_audit_demo_probe(capsys):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "demo_probe.json")
    assert main(["audit", "--probe", path]) == 0
    out = capsys.readouterr().out
    assert "Probe estimates" in out


def test_cli_evaluate_writes_results(tmp_path):
    outdir = str(tmp_path / "out")
    assert main(["evaluate", "--outdir", outdir]) == 0
    with open(os.path.join(outdir, "evaluation.json")) as handle:
        stored = json.load(handle)
    assert stored["aggregates"]["n_scenarios"] == 12


def test_cli_no_command_shows_help(capsys):
    assert main([]) == 2
    out = capsys.readouterr().out
    assert "vote" in out.lower()
