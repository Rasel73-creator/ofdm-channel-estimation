"""Experiment 3: learning R from calibration data, and the two fixes (rank vs shrinkage).

Grid: SNR 0/10/20 dB x S = 8/32/128 calibration channels x clean or noisy calibration.
Rank and shrinkage are each chosen from 16 options on validation channels (equal budgets).
Every number is the mean over 20 calibration sets, scored on the same 400 test channels.
Seeds: calibration 0-19, validation 500 (200 channels), test 999 (400 channels).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # find the recipe book in src/

import matplotlib.pyplot as plt
import numpy as np
from ofdm_tools import (bootstrap_mean, learn_table, lmmse_guess, low_rank_guess, make_dataset,
                        nmse_db, save_results, shrink_table, tune_rank, tune_shrink)

snrs = [0, 10, 20]
sizes = [8, 32, 128]
repeats = 20
methods = ["full", "rank", "shrink"]

def pm(values):
    low, high = bootstrap_mean(values)
    return f"{np.mean(values):.1f} ± {(high - low) / 2:.1f}"

lines = ["## Experiment 3: NMSE (dB) with a learned covariance, mean of 20 calibration sets", "",
         "| calibration | SNR (dB) | S | LS | LMMSE (true R) | learned, full | learned, rank r | learned, shrinkage α |",
         "|---|---|---|---|---|---|---|---|"]
curves = {}          # (calibration, snr) -> {method: [mean score for each S]}
refs = {}            # snr -> (LS, true R)
for snr in snrs:
    noise_var = 10 ** (-snr / 10)
    val_truth, val_obs, _ = make_dataset(seed=500, count=200, noise_var=noise_var)
    test_truth, test_obs, R_true = make_dataset(seed=999, count=400, noise_var=noise_var)
    ls = nmse_db(test_obs, test_truth)
    key = nmse_db(lmmse_guess(test_obs, R_true, noise_var), test_truth)
    refs[snr] = (ls, key)
    for calibration in ["clean", "noisy"]:
        curves[calibration, snr] = {m: [] for m in methods}
        for S in sizes:
            scores = {m: [] for m in methods}
            chosen_r, chosen_a = [], []
            for rep in range(repeats):
                cal_truth, cal_obs, _ = make_dataset(seed=rep, count=S, noise_var=noise_var)
                if calibration == "clean":
                    R_hat = learn_table(cal_truth)                 # simulation privilege (declared)
                else:
                    R_hat = learn_table(cal_obs, noise_var)        # real life: subtract the noise
                r = tune_rank(R_hat, val_obs, val_truth, noise_var)
                a = tune_shrink(R_hat, val_obs, val_truth, noise_var)
                scores["full"].append(nmse_db(lmmse_guess(test_obs, R_hat, noise_var), test_truth))
                scores["rank"].append(nmse_db(low_rank_guess(test_obs, R_hat, noise_var, r), test_truth))
                scores["shrink"].append(nmse_db(lmmse_guess(test_obs, shrink_table(R_hat, a), noise_var),
                                                test_truth))
                chosen_r.append(r)
                chosen_a.append(a)
            for m in methods:
                curves[calibration, snr][m].append(np.mean(scores[m]))
            lines.append(f"| {calibration} | {snr} | {S} | {ls:.1f} | {key:.1f} | {pm(scores['full'])} | "
                         f"{pm(scores['rank'])} (r = {int(np.median(chosen_r))}) | "
                         f"{pm(scores['shrink'])} (α = {np.median(chosen_a):.2f}) |")
        print(f"done: {calibration} calibration, SNR {snr} dB")

lines += ["", "± is half the width of the 95% bootstrap range of the mean (resampling the 20 calibration sets).",
          "r and α are the medians of the values chosen on the validation channels.",
          "Clean calibration and the true validation channels used for scoring are declared simulation privileges."]

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="col")
for row, calibration in enumerate(["clean", "noisy"]):
    for col, snr in enumerate(snrs):
        ax = axes[row, col]
        for m, marker in zip(methods, ["s-", "o-", "^-"]):
            ax.plot(sizes, curves[calibration, snr][m], marker, label=f"learned R, {m}")
        ax.axhline(refs[snr][0], color="grey", linestyle=":", label="LS")
        ax.axhline(refs[snr][1], color="black", linestyle="--", label="LMMSE (true R)")
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.set_title(f"{calibration} calibration, SNR {snr} dB")
        ax.grid(True, alpha=0.3)
        if row == 1:
            ax.set_xlabel("calibration channels S")
        if col == 0:
            ax.set_ylabel("NMSE (dB, lower is better)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save_results("exp3_calibration", lines, fig)