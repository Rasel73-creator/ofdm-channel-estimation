"""Experiment 1: LS vs LMMSE vs rank-8 over SNR, using the true covariance (the reference methods).

Checks the code against theory: LS should score -SNR dB, and LMMSE should match
sum(lambda * s2 / (lambda + s2)) / sum(lambda), where s2 is the noise variance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # find the recipe book in src/

import matplotlib.pyplot as plt
import numpy as np
from ofdm_tools import bootstrap_db, low_rank_guess, lmmse_guess, make_dataset, nmse_db, save_results

snrs = list(range(0, 31, 5))
names = ["LS", "LMMSE (true R)", "rank-8 (true R)"]
scores = {name: [] for name in names}
theory_ls, theory_lmmse = [], []

lines = ["## Experiment 1: NMSE (dB) vs SNR, true covariance, 400 test channels (seed 999)", "",
         "| SNR (dB) | LS | LMMSE (true R) | rank-8 (true R) | theory LS | theory LMMSE |",
         "|---|---|---|---|---|---|"]
for snr in snrs:
    noise_var = 10 ** (-snr / 10)
    truth, obs, R_true = make_dataset(seed=999, count=400, noise_var=noise_var)
    guesses = {"LS": obs,
               "LMMSE (true R)": lmmse_guess(obs, R_true, noise_var),
               "rank-8 (true R)": low_rank_guess(obs, R_true, noise_var, rank=8)}
    cells = []
    for name in names:
        score = nmse_db(guesses[name], truth)
        low, high = bootstrap_db(guesses[name], truth)
        scores[name].append(score)
        cells.append(f"{score:.1f} ± {(high - low) / 2:.1f}")

    lam = np.clip(np.linalg.eigvalsh(R_true), 0, None)            # knob strengths (eigenvalues)
    theory_ls.append(10 * np.log10(noise_var))
    theory_lmmse.append(10 * np.log10(np.sum(lam * noise_var / (lam + noise_var)) / np.sum(lam)))
    lines.append(f"| {snr} | " + " | ".join(cells) + f" | {theory_ls[-1]:.1f} | {theory_lmmse[-1]:.1f} |")

lines += ["", "± is half the width of the 95% bootstrap range (resampling whole test channels).",
          "LMMSE and rank-8 are identical here because the true R has exactly 8 non-zero eigenvalues."]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(snrs, theory_ls, "--", color="grey", label="theory LS")
ax.plot(snrs, theory_lmmse, "--", color="black", label="theory LMMSE")
ax.plot(snrs, scores["LS"], "o", label="LS")
ax.plot(snrs, scores["LMMSE (true R)"], "s", label="LMMSE (true R)")
ax.plot(snrs, scores["rank-8 (true R)"], "x", markersize=9, label="rank-8 (true R)")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("NMSE (dB, lower is better)")
ax.set_title("Reference methods vs SNR (N = 64, 8 paths)")
ax.grid(True, alpha=0.3)
ax.legend()
save_results("exp1_snr_sweep", lines, fig)