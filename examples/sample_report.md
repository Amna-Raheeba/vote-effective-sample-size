# Self-consistency vote audit

## 1. Probe estimates

- Individual accuracy (p_hat): 0.767
- Error correlation, agreement estimator (rho_hat): 0.425
- Error correlation, variance estimator: 0.435
- Pairwise agreement: 0.794
- Agreement expected under independence: 0.643
- Share of agreement that is bias-driven: 19.1%
- Accuracy ceiling at any sample count: 0.834

## 2. What the votes are worth

| Samples | Effective votes (n_eff) | Predicted accuracy | Predicted gain | Naive accuracy | Naive gain |
|---|---|---|---|---|---|
| 8 | 2.01 | 0.815 | 0.079 | 0.963 | 0.227 |
| 16 | 2.17 | 0.824 | 0.088 | 0.994 | 0.258 |
| 32 | 2.26 | 0.829 | 0.093 | 1.000 | 0.264 |
| 64 | 2.31 | 0.831 | 0.095 | 1.000 | 0.264 |

## 3. Recommendation

Voting helps but is discounted: 64 samples are worth 2.3 independent votes (rho = 0.425). Predicted gain 0.095, against the 0.264 the naive independence assumption promises.

