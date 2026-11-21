## Experiment 5: computational cost (one processor core)

Computer: Intel64 Family 6 Model 142 Stepping 9, GenuineIntel, Python 3.14.8, numpy 2.5.4, matplotlib 3.11.2

| N | setup, full LMMSE (ms) | setup, rank-8 (ms) | per block, full LMMSE (µs) | per block, rank-8 (µs) |
|---|---|---|---|---|
| 64 | 0.47 | 1.01 | 1.42 | 0.63 |
| 128 | 2.78 | 6.58 | 5.72 | 1.73 |
| 256 | 12.97 | 24.12 | 12.08 | 2.62 |
| 512 | 52.74 | 175.35 | 46.57 | 5.44 |
| 1024 | 342.48 | 1898.79 | 172.19 | 10.61 |

From N = 256 to N = 1024 (4x): setup grew 26x (full) and 79x (rank-8), rule of thumb 64x; per block grew 14x (full, rule 16x) and 4.1x (rank-8, rule 4x).

### Summary at N = 64, SNR 10 dB (true R, 400 test channels)

| method | rank | NMSE (dB) | setup (ms) | per block (µs) |
|---|---|---|---|---|
| LS | - | -9.9 | none | none |
| LMMSE | 64 (full) | -19.0 | 0.47 | 1.42 |
| rank-8 | 8 | -19.0 | 1.01 | 0.63 |
