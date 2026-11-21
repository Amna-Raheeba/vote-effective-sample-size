"""Data models for the vote effective-sample-size audit.

A *population* is a vote-generating process: every sample it produces is
correct with probability ``p``, but the samples share a latent bias factor,
so their errors are correlated with a known strength ``rho_latent``.

A *probe set* is a small labelled set of cases, each with several samples,
used to estimate that correlation before paying for a full vote.
"""

from dataclasses import dataclass, field

MAX_RHO_LATENT = 0.999


@dataclass
class Population:
    """A vote population with planted shared bias.

    p: probability that a single sample is correct (0 < p < 1).
    rho_latent: correlation of the latent Gaussian factor shared by all
        samples of one case (0 <= rho_latent <= 0.999). 0 means the samples
        are independent; values near 1 mean they mostly succeed or fail
        together.
    seed: bookkeeping seed for the population definition.
    """

    p: float
    rho_latent: float
    seed: int = 7

    def __post_init__(self):
        if not 0.0 < float(self.p) < 1.0:
            raise ValueError("p must be strictly between 0 and 1, got %r" % (self.p,))
        if not 0.0 <= float(self.rho_latent) <= MAX_RHO_LATENT:
            raise ValueError(
                "rho_latent must be between 0 and %s, got %r"
                % (MAX_RHO_LATENT, self.rho_latent)
            )


@dataclass
class ProbeSet:
    """Labelled probe votes: one row per case, one bool per sample.

    ``votes[j][i]`` is True when sample i answered case j correctly.
    """

    votes: list

    def __post_init__(self):
        if not self.votes:
            raise ValueError("probe set must contain at least one case")
        width = len(self.votes[0])
        if width < 2:
            raise ValueError("each case needs at least 2 samples, got %d" % width)
        for row in self.votes:
            if len(row) != width:
                raise ValueError("probe rows must all have the same number of samples")
            for value in row:
                if not isinstance(value, bool):
                    raise ValueError("probe votes must be booleans, got %r" % (value,))

    @property
    def n_cases(self):
        return len(self.votes)

    @property
    def n_samples(self):
        return len(self.votes[0])

    def correct_counts(self):
        return [sum(1 for value in row if value) for row in self.votes]

    def to_dict(self):
        return {"votes": [list(row) for row in self.votes]}

    @classmethod
    def from_dict(cls, payload):
        if "votes" not in payload:
            raise ValueError("probe payload must contain a 'votes' key")
        return cls(votes=[[bool(value) for value in row] for row in payload["votes"]])


@dataclass
class AuditRow:
    """Predicted vote performance at one target sample count."""

    n: int
    n_eff: float
    pred_accuracy: float
    pred_gain: float
    naive_accuracy: float
    naive_gain: float


@dataclass
class AuditResult:
    """The full audit of one probe set."""

    p_hat: float
    rho_hat: float
    rho_hat_variance: float
    agreement: float
    independent_agreement: float
    bias_share: float
    ceiling: float
    min_gain: float
    rows: list = field(default_factory=list)
    recommendation: str = ""

    def row_for(self, n):
        for row in self.rows:
            if row.n == n:
                return row
        return None
