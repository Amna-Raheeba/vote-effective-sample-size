import pytest

from vote_eff.models import Population, ProbeSet


def test_population_valid():
    pop = Population(p=0.7, rho_latent=0.6, seed=3)
    assert pop.p == 0.7
    assert pop.rho_latent == 0.6
    assert pop.seed == 3


def test_population_default_seed():
    assert Population(p=0.6, rho_latent=0.0).seed == 7


@pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.1, 1.5])
def test_population_rejects_bad_p(bad_p):
    with pytest.raises(ValueError):
        Population(p=bad_p, rho_latent=0.3)


@pytest.mark.parametrize("bad_rho", [-0.1, 1.0, 2.0])
def test_population_rejects_bad_rho(bad_rho):
    with pytest.raises(ValueError):
        Population(p=0.7, rho_latent=bad_rho)


def test_probe_properties():
    probe = ProbeSet(votes=[[True, False, True], [False, False, True]])
    assert probe.n_cases == 2
    assert probe.n_samples == 3
    assert probe.correct_counts() == [2, 1]


def test_probe_rejects_empty():
    with pytest.raises(ValueError):
        ProbeSet(votes=[])


def test_probe_rejects_single_sample():
    with pytest.raises(ValueError):
        ProbeSet(votes=[[True], [False]])


def test_probe_rejects_ragged():
    with pytest.raises(ValueError):
        ProbeSet(votes=[[True, False], [True]])


def test_probe_rejects_non_bool():
    with pytest.raises(ValueError):
        ProbeSet(votes=[[True, 1], [False, True]])


def test_probe_dict_round_trip():
    probe = ProbeSet(votes=[[True, False], [False, False]])
    clone = ProbeSet.from_dict(probe.to_dict())
    assert clone.votes == probe.votes


def test_probe_from_dict_requires_votes_key():
    with pytest.raises(ValueError):
        ProbeSet.from_dict({"rows": [[True, False]]})
