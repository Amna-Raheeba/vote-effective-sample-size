"""Probe estimators: from observed votes to an error correlation.

The key identity: in a binary vote (correct answer vs the shared wrong
attractor), two samples agree exactly when they are both correct or both
wrong. If their correctness indicators have mean p and correlation rho,
then

    P(agree) = p^2 + (1 - p)^2 + 2 * rho * p * (1 - p),

so the excess agreement over the independence level p^2 + (1 - p)^2 is a
direct measurement of rho. A second, independent estimator uses the
variance of the per-case correct counts, which the same correlation
inflates by the design effect 1 + (n - 1) * rho.
"""


def _clip01(value):
    return max(0.0, min(1.0, value))


def estimate_accuracy(probe):
    """Overall fraction of correct votes in the probe."""
    correct = 0
    total = 0
    for row in probe.votes:
        for value in row:
            total += 1
            if value:
                correct += 1
    return correct / total


def pairwise_agreement(probe):
    """Mean fraction of within-case sample pairs that agree."""
    agree = 0
    pairs = 0
    for row in probe.votes:
        k = sum(1 for value in row if value)
        m = len(row)
        agree += k * (k - 1) // 2 + (m - k) * (m - k - 1) // 2
        pairs += m * (m - 1) // 2
    return agree / pairs


def independent_agreement(p):
    """Agreement expected if the samples were independent."""
    return p * p + (1.0 - p) * (1.0 - p)


def estimate_rho_agreement(probe):
    """Error correlation from excess pairwise agreement (primary estimator)."""
    p = estimate_accuracy(probe)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    observed = pairwise_agreement(probe)
    rho = (observed - independent_agreement(p)) / (2.0 * p * (1.0 - p))
    return _clip01(rho)


def estimate_rho_variance(probe):
    """Error correlation from the variance of per-case correct counts."""
    p = estimate_accuracy(probe)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    n = probe.n_samples
    if n < 2 or probe.n_cases < 2:
        return 0.0
    counts = probe.correct_counts()
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / (len(counts) - 1)
    rho = (var / (n * p * (1.0 - p)) - 1.0) / (n - 1)
    return _clip01(rho)


def effective_sample_size(n, rho):
    """How many independent votes n correlated votes are worth.

    The design-effect formula: the variance of the mean of n equicorrelated
    votes equals the variance of n_eff independent votes, with
    n_eff = n / (1 + (n - 1) * rho).
    """
    if n < 1:
        raise ValueError("n must be at least 1, got %r" % (n,))
    rho = _clip01(rho)
    return n / (1.0 + (n - 1) * rho)


def agreement_decomposition(probe):
    """Split observed agreement into independent and bias-driven parts."""
    p = estimate_accuracy(probe)
    observed = pairwise_agreement(probe)
    indep = independent_agreement(p)
    excess = max(0.0, observed - indep)
    share = excess / observed if observed > 0.0 else 0.0
    return {
        "p_hat": p,
        "agreement": observed,
        "independent_agreement": indep,
        "excess_agreement": excess,
        "bias_share": share,
    }
