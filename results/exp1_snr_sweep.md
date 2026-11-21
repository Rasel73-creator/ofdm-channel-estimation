## Experiment 1: NMSE (dB) vs SNR, true covariance, 400 test channels (seed 999)

| SNR (dB) | LS | LMMSE (true R) | rank-8 (true R) | theory LS | theory LMMSE |
|---|---|---|---|---|---|
| 0 | 0.1 ± 0.2 | -10.1 ± 0.2 | -10.1 ± 0.2 | 0.0 | -10.2 |
| 5 | -4.9 ± 0.2 | -14.4 ± 0.2 | -14.4 ± 0.2 | -5.0 | -14.5 |
| 10 | -9.9 ± 0.2 | -19.0 ± 0.2 | -19.0 ± 0.2 | -10.0 | -19.2 |
| 15 | -14.9 ± 0.2 | -23.9 ± 0.2 | -23.9 ± 0.2 | -15.0 | -24.1 |
| 20 | -19.9 ± 0.2 | -28.9 ± 0.2 | -28.9 ± 0.2 | -20.0 | -29.0 |
| 25 | -24.9 ± 0.2 | -33.9 ± 0.2 | -33.9 ± 0.2 | -25.0 | -34.0 |
| 30 | -29.9 ± 0.2 | -38.9 ± 0.2 | -38.9 ± 0.2 | -30.0 | -39.0 |

± is half the width of the 95% bootstrap range (resampling whole test channels).
LMMSE and rank-8 are identical here because the true R has exactly 8 non-zero eigenvalues.
