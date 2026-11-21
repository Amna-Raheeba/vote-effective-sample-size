from vote_eff.audit import run_audit
from vote_eff.models import Population, ProbeSet
from vote_eff.sampler import generate_probe


def test_audit_rows_cover_targets_sorted():
    probe = generate_probe(Population(p=0.7, rho_latent=0.6), 48, 16, 8)
    result = run_audit(probe, targets=(32, 8, 16))
    assert [row.n for row in result.rows] == [8, 16, 32]


def test_audit_biased_probe_discounts_votes():
    probe = generate_probe(Population(p=0.7, rho_latent=0.6), 48, 16, 8)
    result = run_audit(probe)
    row32 = result.row_for(32)
    assert row32 is not None
    assert row32.n_eff < 8.0
    assert result.rho_hat > 0.15
    assert result.ceiling < 1.0


def test_audit_independent_probe_keeps_votes():
    probe = generate_probe(Population(p=0.7, rho_latent=0.0), 200, 16, 21)
    result = run_audit(probe)
    assert result.rho_hat < 0.10
    assert result.row_for(32).n_eff > 20.0


def test_audit_naive_gain_dominates_tool_gain():
    probe = generate_probe(Population(p=0.7, rho_latent=0.6), 48, 16, 8)
    result = run_audit(probe)
    for row in result.rows:
        assert row.naive_gain >= row.pred_gain


def test_audit_degenerate_probe_does_not_crash():
    probe = ProbeSet(votes=[[True, True, True], [True, True, True]])
    result = run_audit(probe)
    assert result.p_hat == 1.0
    assert result.rho_hat == 0.0
    assert result.recommendation


def test_audit_below_chance_recommendation():
    probe = ProbeSet(votes=[[False, False, True], [False, True, False]])
    result = run_audit(probe)
    assert result.p_hat < 0.5
    assert "Do not vote" in result.recommendation


def test_audit_bias_share_in_unit_interval():
    probe = generate_probe(Population(p=0.8, rho_latent=0.9), 48, 16, 8)
    result = run_audit(probe)
    assert 0.0 <= result.bias_share <= 1.0
    assert result.recommendation
