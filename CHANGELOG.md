# Changelog

## 0.1.0 — 2026-11-21

- Initial release.
- Seeded vote sampler with a planted shared-bias (common latent factor)
  component, so the error correlation of a vote population is known exactly.
- Analytic ground truth: indicator correlation via the bivariate normal CDF,
  and the design-effect effective sample size n_eff = n / (1 + (n-1) * rho).
- Probe estimators: pairwise-agreement correlation and count-variance
  correlation, plus an agreement decomposition (independent vs bias-driven).
- Majority-vote accuracy predictor (normal approximation at n_eff), an
  accuracy ceiling under correlation, and a samples-needed solver.
- Headline evaluation: 12 scenarios (accuracy x bias strength), predicted vs
  realised majority-vote gain against the naive independence baseline.
- Streamlit dashboard (3 tabs), CLI, notebooks, and a reproducibility script
  with a headline-drift guard.
