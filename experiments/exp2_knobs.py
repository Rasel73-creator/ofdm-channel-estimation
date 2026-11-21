"""Experiment 2: the hidden knobs. Eigenvalues of R, and NMSE vs the number of knobs kept (rank).

True R: the error drops until rank 8, then stays flat (matches theory).
Learned R (128 noisy calibration channels): extra knobs are made of noise, so the curve
rises again after rank 8. This is the bias-variance trade-off.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # find the recipe book in src/

import matplotlib.pyplot as plt
import numpy as np
from ofdm_tools import learn_table, low_rank_guess, make_dataset, nmse_db, save_results

noise_var = 0.1                                       # SNR 10 dB
S = 128                                               # noisy calibration channels for the learned table
ranks = list(range(1, 17)) + [24, 32, 48, 64]

test_truth, test_obs, R_true = make_dataset(seed=999, count=400, noise_var=noise_var)
lam = np.clip(np.sort(np.linalg.eigvalsh(R_true))[::-1], 0, None)       # true knob strengths, strongest first

true_curve = [nmse_db(low_rank_guess(test_obs, R_true, noise_var, r), test_truth) for r in ranks]
theory = [10 * np.log10((np.sum(lam[:r] * noise_var / (lam[:r] + noise_var)) + np.sum(lam[r:])) / np.sum(lam))
          for r in ranks]

learned_curves = []
for rep in range(20):                                 # 20 different calibration sets
    _, cal_obs, _ = make_dataset(seed=rep, count=S, noise_var=noise_var)
    R_learned = learn_table(cal_obs, noise_var)
    learned_curves.append([nmse_db(low_rank_guess(test_obs, R_learned, noise_var, r), test_truth)
                           for r in ranks])
learned_curve = np.mean(learned_curves, axis=0)

_, cal_obs, _ = make_dataset(seed=0, count=S, noise_var=noise_var)
lam_learned = np.sort(np.linalg.eigvalsh(learn_table(cal_obs, noise_var)))[::-1]

lines = ["## Experiment 2: NMSE (dB) vs number of knobs kept (rank r), SNR 10 dB, 400 test channels", "",
         "| r | true R | theory (true R) | learned R, 128 noisy channels (mean of 20 sets) |",
         "|---|---|---|---|"]
for r, a, b, c in zip(ranks, true_curve, theory, learned_curve):
    if r in [1, 2, 4, 6, 8, 10, 12, 16, 32, 64]:
        lines.append(f"| {r} | {a:.1f} | {b:.1f} | {c:.1f} |")
lines += ["", "Strongest 12 eigenvalues of the true R:    " + ", ".join(f"{v:.2f}" for v in lam[:12]),
          "", "Strongest 12 eigenvalues of one learned R: " + ", ".join(f"{v:.2f}" for v in lam_learned[:12]),
          "", "The test channels are only used to draw this curve. Experiment 3 chooses r on separate validation channels."]

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
k = np.arange(1, 17)
left.semilogy(k[:8], lam[:8], "o", label="true R (the other 56 are zero)")
left.semilogy(k, lam_learned[:16], "s", label=f"learned R ({S} noisy channels)")
left.set_xticks(k)
left.set_xlabel("knob number j (strongest first)")
left.set_ylabel("eigenvalue (knob strength)")
left.set_title("Knob strengths: 8 real knobs, then noise")
left.grid(True, which="both", alpha=0.3)
left.legend()

right.plot(ranks, theory, "--", color="black", label="theory (true R)")
right.plot(ranks, true_curve, "o", label="true R")
right.plot(ranks, learned_curve, "s-", label=f"learned R ({S} noisy channels)")
right.axhline(nmse_db(test_obs, test_truth), color="grey", linestyle=":", label="LS")
right.set_xscale("log", base=2)
right.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
right.set_xlabel("number of knobs kept (rank r)")
right.set_ylabel("NMSE (dB, lower is better)")
right.set_title("NMSE vs rank (SNR 10 dB)")
right.grid(True, which="both", alpha=0.3)
right.legend()
fig.tight_layout()
save_results("exp2_knobs", lines, fig)