"""vote_eff: effective sample size for self-consistency votes.

Converts observed agreement between samples into an effective number of
independent votes (n_eff) under correlated error, and predicts the
majority-vote accuracy gain from additional samples before paying for them.

Author: Amna Raheeba
"""

from .audit import run_audit
from .estimators import effective_sample_size, estimate_rho_agreement
from .models import AuditResult, AuditRow, Population, ProbeSet
from .predict import accuracy_ceiling, predicted_gain, samples_needed

__version__ = "0.1.0"

__all__ = [
    "AuditResult",
    "AuditRow",
    "Population",
    "ProbeSet",
    "accuracy_ceiling",
    "effective_sample_size",
    "estimate_rho_agreement",
    "predicted_gain",
    "run_audit",
    "samples_needed",
]
