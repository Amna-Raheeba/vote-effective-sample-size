"""Run the audit: probe votes in, a priced vote decision out."""

from .estimators import (
    agreement_decomposition,
    effective_sample_size,
    estimate_rho_agreement,
    estimate_rho_variance,
)
from .models import AuditResult, AuditRow
from .predict import accuracy_ceiling, predicted_accuracy, predicted_gain

DEFAULT_TARGETS = (8, 16, 32, 64)


def _recommend(p_hat, rho_hat, rows, min_gain):
    if not rows:
        return "No target sample counts were requested."
    biggest = rows[-1]
    if p_hat <= 0.5:
        return (
            "Do not vote: individual accuracy is at or below chance "
            "(p = %.3f), so a majority vote cannot help." % p_hat
        )
    if rho_hat < 0.05:
        return (
            "Votes are nearly independent (rho = %.3f): self-consistency is "
            "worth close to its nominal sample count, and %d samples are "
            "worth %.1f independent votes."
            % (rho_hat, biggest.n, biggest.n_eff)
        )
    if biggest.pred_gain < min_gain:
        return (
            "Do not pay for more samples: at rho = %.3f, %d samples are worth "
            "only %.1f independent votes and the predicted gain is %.3f, "
            "below the %.3f minimum worth paying for."
            % (rho_hat, biggest.n, biggest.n_eff, biggest.pred_gain, min_gain)
        )
    return (
        "Voting helps but is discounted: %d samples are worth %.1f "
        "independent votes (rho = %.3f). Predicted gain %.3f, against the "
        "%.3f the naive independence assumption promises."
        % (biggest.n, biggest.n_eff, rho_hat, biggest.pred_gain, biggest.naive_gain)
    )


def run_audit(probe, targets=DEFAULT_TARGETS, min_gain=0.02):
    """Audit one probe set and predict vote performance at each target n."""
    decomposition = agreement_decomposition(probe)
    p_hat = decomposition["p_hat"]
    rho_hat = estimate_rho_agreement(probe)
    rho_var = estimate_rho_variance(probe)
    rows = []
    for n in sorted(set(int(t) for t in targets)):
        rows.append(
            AuditRow(
                n=n,
                n_eff=effective_sample_size(n, rho_hat),
                pred_accuracy=predicted_accuracy(p_hat, rho_hat, n),
                pred_gain=predicted_gain(p_hat, rho_hat, n),
                naive_accuracy=predicted_accuracy(p_hat, 0.0, n),
                naive_gain=predicted_gain(p_hat, 0.0, n),
            )
        )
    return AuditResult(
        p_hat=p_hat,
        rho_hat=rho_hat,
        rho_hat_variance=rho_var,
        agreement=decomposition["agreement"],
        independent_agreement=decomposition["independent_agreement"],
        bias_share=decomposition["bias_share"],
        ceiling=accuracy_ceiling(p_hat, rho_hat),
        min_gain=min_gain,
        rows=rows,
        recommendation=_recommend(p_hat, rho_hat, rows, min_gain),
    )
