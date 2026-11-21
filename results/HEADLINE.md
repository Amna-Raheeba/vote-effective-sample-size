# Headline results

Self-consistency votes audited on 12 synthetic scenarios with known planted bias (seeded, analytic ground truth).

- Showcase (p = 0.70, latent bias 0.9): 32 samples are worth 1.40 independent votes in truth; the audit estimates 1.40 from a 48 x 16 probe.
- At that bias level the realised 32-vote gain is 0.005; this tool predicts 0.022, the naive independence assumption promises 0.339.
- Gain-prediction MAE across all scenarios and target sizes: 0.0449 for this tool vs 0.1661 for the naive independence baseline.
- At high bias (rho_latent >= 0.6) the naive baseline over-promises by +0.2356 accuracy points of gain on average; this tool's overall signed error is +0.0392.
- Mean relative error of the recovered n_eff at n = 32, biased scenarios only: 12.5%.

Caveat: all votes are synthetic, drawn from the tool's own seeded generative model; the experiment measures estimator recovery and prediction accuracy under that model, not live LLM behaviour.
