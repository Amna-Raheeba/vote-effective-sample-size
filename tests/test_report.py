from vote_eff.audit import run_audit
from vote_eff.evaluate import run_evaluation
from vote_eff.models import Population
from vote_eff.report import (
    audit_report_markdown,
    evaluation_summary_markdown,
    headline_markdown,
)
from vote_eff.sampler import generate_probe

SMALL_CONFIG = {
    "p_values": [0.70],
    "rho_values": [0.9],
    "probe_cases": 24,
    "probe_samples": 12,
    "realised_cases": 400,
    "targets": [32],
    "base_seed": 7,
}


def test_audit_report_structure():
    probe = generate_probe(Population(p=0.7, rho_latent=0.6), 48, 16, 8)
    result = run_audit(probe)
    text = audit_report_markdown(result)
    assert "# Self-consistency vote audit" in text
    assert "## 1. Probe estimates" in text
    assert "## 2. What the votes are worth" in text
    assert "## 3. Recommendation" in text
    assert result.recommendation in text


def test_audit_report_contains_numbers():
    probe = generate_probe(Population(p=0.7, rho_latent=0.6), 48, 16, 8)
    result = run_audit(probe)
    text = audit_report_markdown(result)
    assert "%.3f" % result.rho_hat in text
    assert "| 32 |" in text


def test_evaluation_summary_mentions_scenarios():
    evaluation = run_evaluation(SMALL_CONFIG)
    text = evaluation_summary_markdown(evaluation)
    assert "Scenarios: 1" in text
    assert "Per-scenario detail" in text


def test_headline_mentions_showcase_and_caveat():
    evaluation = run_evaluation(SMALL_CONFIG)
    text = headline_markdown(evaluation)
    assert "Showcase" in text
    assert "independent votes" in text
    assert "synthetic" in text
