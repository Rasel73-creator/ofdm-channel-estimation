# Low-rank OFDM channel estimation (Track T)

Final project for **Computational Methods and Algorithms**, by Mohammad Rasel.

The project estimates an OFDM channel from noisy pilots, following
O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson and P. O. Börjesson,
"OFDM Channel Estimation by Singular Value Decomposition", *IEEE Transactions on Communications*,
46(7), 931–939, 1998. https://doi.org/10.1109/26.701321

Three estimators are compared:

- **LS** (least squares): trust the pilot measurement as it is, ĥ = z.
- **LMMSE**: ĥ = R(R + σ²I)⁻¹z, which uses how the subcarriers vary together (the covariance R).
- **Rank-r LMMSE**: keep only the r strongest eigen-directions of R and drop the rest.

## What's in this repository

| Folder | Contents |
|---|---|
| `src/` | `ofdm_tools.py`: channel model, estimators, covariance learning, tuning and scoring |
| `experiments/` | one script for each result in the report |
| `results/` | the tables (`.md`) and figures (`.png`) that the experiments save |
| `playground/` | my step-by-step learning lessons, kept as a record of how I built the project (not needed to reproduce the results) |

## How to run

1. Create a virtual environment: `python -m venv .venv`, then activate it (Windows: `.venv\Scripts\activate`; macOS/Linux: `source .venv/bin/activate`).
2. Install the packages: `pip install -r requirements.txt`
3. Run any experiment, for example: `python experiments/exp1_snr_sweep.py`

Each script prints its table, then saves the table and its figure in `results/`.

| Script | What it shows |
|---|---|
| `exp1_snr_sweep.py` | LS vs LMMSE vs rank-8 over SNR with the true covariance, compared with theory |
| `exp2_knobs.py` | eigenvalues of R, and NMSE vs rank (the bias–variance trade-off) |
| `exp3_calibration.py` | R learned from S = 8 / 32 / 128 clean or noisy calibration channels; rank truncation vs shrinkage, both tuned on validation data |
| `exp4_mismatch.py` | stress test: the channel changes from 8 to 12 paths; robust design |
| `exp5_cost.py` | setup time vs per-block time for N = 64 to 1024 subcarriers (one CPU core) |

## Experimental protocol

- **Model** (from the course guide): N = 64 subcarriers, 8 paths at delays ℓ + 0.25 samples with power ∝ exp(−ℓ/2), unit pilots, so z = h + w. SNR = 1/σ².
- **Separate random draws:** calibration seeds 0–19 (S channels each), validation seed 500 (200 channels), test seed 999 (400 channels).
- **Tuning:** the rank (1 to 16) and the shrinkage α (16 values from 0 to 1) are chosen on the validation channels, so both fixes get the same information and the same tuning budget. Ties go to the simpler setting.
- **Uncertainty:** 95% bootstrap ranges that resample whole channels (test channels in Experiments 1 and 4, whole calibration sets in Experiment 3).
- **Simulation privileges (declared):** in the "clean calibration" case, R is learned from the true calibration channels; validation NMSE is computed with the true validation channels. No method ever sees the true test channels.

## Results at a glance (SNR 10 dB unless stated)

- Using the covariance gains about **9 dB** over LS at every SNR (−19.0 vs −9.9 dB). This matches the simple picture: the channel lives in 8 of the 64 directions, and 10·log10(64/8) ≈ 9 dB. Rank-8 is exactly as accurate, because the true R has only 8 non-zero eigenvalues.
- The simulations agree with theory within 0.2 dB (Experiments 1 and 2).
- A covariance learned from noisy calibration needs a fix. With S = 8, shrinkage improves −7.8 to −11.6 dB; with S = 128, rank truncation improves −16.6 to −17.3 dB (Experiment 3).
- When the channel changes, the old covariance hits an error floor near −12.5 dB and becomes worse than LS above about 12 dB SNR. The robust design never breaks, but costs about 3.5–4 dB when the channel has not changed (Experiment 4).
- Per block, the rank-8 work grows like N·r instead of N². Finding the eigenvectors costs more than building the full LMMSE matrix, but it is done only once (Experiment 5).

## Scope

Synchronisation, sparse pilots and the paper's full modulation experiments are outside the scope of this project.

## Tested with

Windows, Python 3.14. The exact numpy and matplotlib versions are printed at the top of `results/exp5_cost.md`.