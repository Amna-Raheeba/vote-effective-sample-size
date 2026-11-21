# Vote Effective Sample Size

An agreement-discount audit for self-consistency voting: it converts the
agreement between samples into an **effective number of independent votes
(n_eff)** under correlated error, and predicts the majority-vote accuracy
gain from additional samples *before* you pay for them.

## 1. What it is

Self-consistency voting assumes each sample is a fresh, independent vote.
It is not. Samples from one model share biases — a positional preference,
a misread phrase, a wrong attractor — so their errors are correlated, and
32 samples may be worth only a handful of independent votes. The naive
arithmetic ("32 votes at 70% each gives ~99%") systematically over-promises.

This tool is the pre-flight check. Give it a small **labelled probe** (a
few dozen cases, each answered several times) and it returns:

1. the estimated error correlation rho between samples,
2. an agreement decomposition: how much of the observed agreement is
   independent agreement and how much is bias-driven,
3. the effective sample size n_eff at any target sample count,
4. the predicted majority-vote accuracy and gain at that count, next to
   the naive independence prediction,
5. the **accuracy ceiling** — the best vote accuracy reachable at any
   sample count while the correlation holds — and the number of samples
   needed to reach a target accuracy, or a plain statement that the
   target is unreachable by voting alone.

Everything is offline, deterministic and seeded. Pure Python standard
library; no API keys, no model downloads, no network at runtime.

## 2. The novel mechanism

Agreement is converted into an information quantity, not reported as a
percentage. In a binary vote (correct answer vs the shared wrong
attractor), two samples agree exactly when both are correct or both are
wrong, so with individual accuracy p and error correlation rho:

```
P(agree) = p^2 + (1 - p)^2 + 2 * rho * p * (1 - p)
```

The excess agreement over the independence level `p^2 + (1 - p)^2` is a
direct measurement of rho. From rho, the classical design effect gives

```
n_eff = n / (1 + (n - 1) * rho),
```

and majority-vote accuracy is predicted with the normal approximation
evaluated at n_eff. A second, independent estimator (the variance of the
per-case correct counts, which the same correlation inflates) cross-checks
the first.

To validate the instrument, the sampler plants a **shared-bias component**
with known strength: each case draws one latent factor shared by all its
samples, so the true indicator correlation is known analytically through
the bivariate normal CDF, and predicted gains are checked against realised
majority-vote gains from large seeded simulations.

## 3. Prior art and the differentiator

- "Self-Consistency Falls Short!" (TACL) shows positional bias makes
  self-consistency degrade on long contexts.
- "Reasoning Concentrates Errors, and Self-Consistency Never Notices" and
  "When Self-Consistency Backfires" (COLM 2026 workshop) document cases
  where voting fails on live models.

All three are live-model empirical studies. None provides an offline audit
tool, an n_eff metric for a vote, or a predictor of vote gain from a
measurable correlation estimate. The differentiator is the deployable
pre-flight check: estimate the correlation on a small labelled probe,
predict whether voting will help, and learn the accuracy ceiling — before
paying for 32 samples per question.

## 4. Measured headline results

Seeded evaluation, 12 scenarios (individual accuracy p in {0.60, 0.70,
0.80} x latent bias in {0.0, 0.3, 0.6, 0.9}); the audit sees only a
48-case x 16-sample probe per scenario; realised gains come from 3,000
simulated cases per scenario. Full detail in `results/`.

- **Showcase** (p = 0.70, latent bias 0.9): 32 samples are worth **1.40
  independent votes** in truth; the audit estimates **1.40** from the
  probe. The realised 32-vote gain is **0.005**; this tool predicts
  **0.022**; the naive independence assumption promises **0.339**.
- **Gain-prediction MAE** across all scenarios and target sizes (16, 32,
  64 samples): **0.0449** for this tool vs **0.1661** for the naive
  independence baseline.
- At high bias (latent >= 0.6) the naive baseline over-promises by
  **+0.2356 accuracy points** of gain on average — gain that never
  arrives. This tool's overall signed error is +0.0392.
- Mean relative error of the recovered n_eff at n = 32, biased scenarios
  only: **12.5%**.

## 5. Quick start

```bash
pip install -r requirements.txt

python -m vote_eff demo                          # audit a built-in demo population
python -m vote_eff audit --probe examples/demo_probe.json
python -m vote_eff evaluate                      # regenerate results/
python examples/run_audit.py                     # same audit via the example script
streamlit run app.py                             # dashboard (3 tabs)
```

Probe format: JSON with one key, `votes`, a list of cases; each case is a
list of booleans, `true` when that sample answered the case correctly
(see `examples/demo_probe.json`).

## 6. How it works

1. **Estimate** — from the probe: individual accuracy p_hat, pairwise
   agreement, and rho from excess agreement (cross-checked by the
   count-variance estimator).
2. **Decompose** — split observed agreement into the part independence
   would produce anyway and the bias-driven excess; report the share.
3. **Discount** — convert any nominal sample count n into n_eff with the
   design-effect formula.
4. **Predict** — majority-vote accuracy from the normal approximation at
   n_eff; gain is measured against the same approximation at n = 1, so
   the comparison with the naive baseline (rho = 0) is like for like.
5. **Decide** — compare the predicted gain at your budget against a
   declared minimum worth paying for; report the ceiling and, when a
   target accuracy is unreachable under the estimated correlation, say
   so instead of selling more samples.

## 7. Repository layout

```
vote_eff/            the package (sampler, estimators, predict, audit,
                     evaluate, report, cli)
data/                probe sets for all 12 evaluation scenarios
tests/               pytest suite (77 tests)
notebooks/           01_demo_run.py, 02_evaluation_experiment.py
examples/            demo probe, runner script, sample report
scripts/             build_cases.py, reproduce_results.py (drift guard)
results/             evaluation.json/.csv, evaluation_summary.md, HEADLINE.md
app.py               Streamlit dashboard (Audit / Prediction / Evaluation)
```

## 8. Honest limits

- **Synthetic votes only.** Every number above is measured on votes drawn
  from this tool's own seeded generative model (a shared latent Gaussian
  factor). The experiment validates estimator recovery and prediction
  under that model; it does not measure any live LLM.
- **Binary votes.** The agreement identity assumes wrong samples land on
  one shared wrong answer. With several distinct wrong answers, agreement
  understates correlation and n_eff is optimistic.
- **Homogeneous population.** One accuracy p and one correlation rho for
  all samples and cases. Mixed-difficulty case sets blur both estimates.
- **Probe noise is real.** A 48 x 16 probe pins rho to roughly +/-0.08;
  n_eff is most sensitive to that error at low correlation (one unbiased
  scenario recovered n_eff 23.0 against a true 32.0). Use a bigger probe
  when the decision is expensive.
- **The predictor still slightly over-promises on average** (signed error
  +0.0392): the normal approximation at n_eff is an approximation, and the
  residual bias is reported, not hidden.
- The accuracy ceiling assumes the correlation is a property of the
  population, not of the sample count; if extra samples are drawn with a
  different prompt or temperature, re-probe.

## 9. Reproducing the results

```bash
python scripts/build_cases.py        # regenerate data/ and the demo probe
python -m vote_eff evaluate          # regenerate results/
python scripts/reproduce_results.py  # re-run and check for headline drift
pytest                               # full test suite
```

## 10. License

MIT. Author: Amna Raheeba.
