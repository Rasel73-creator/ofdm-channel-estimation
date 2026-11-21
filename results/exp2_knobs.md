## Experiment 2: NMSE (dB) vs number of knobs kept (rank r), SNR 10 dB, 400 test channels

| r | true R | theory (true R) | learned R, 128 noisy channels (mean of 20 sets) |
|---|---|---|---|
| 1 | -2.1 | -2.2 | -2.1 |
| 2 | -4.3 | -4.4 | -4.2 |
| 4 | -9.0 | -9.0 | -8.8 |
| 6 | -13.9 | -13.8 | -13.1 |
| 8 | -19.0 | -19.2 | -17.3 |
| 10 | -19.0 | -19.2 | -17.1 |
| 12 | -19.0 | -19.2 | -17.0 |
| 16 | -19.0 | -19.2 | -16.7 |
| 32 | -19.0 | -19.2 | -16.6 |
| 64 | -19.0 | -19.2 | -16.6 |

Strongest 12 eigenvalues of the true R:    25.65, 15.56, 9.44, 5.72, 3.47, 2.11, 1.28, 0.77, 0.00, 0.00, 0.00, 0.00

Strongest 12 eigenvalues of one learned R: 24.37, 14.85, 8.64, 5.71, 3.37, 1.88, 1.24, 0.81, 0.15, 0.13, 0.12, 0.10

The test channels are only used to draw this curve. Experiment 3 chooses r on separate validation channels.
