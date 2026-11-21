"""Experiment 4: stress test. What if the room changes after we learned it?

The old-room table is the true R of the nominal room (8 paths, decay 2), as if learned perfectly.
The changed room has 12 paths with slower decay (decay 3). The robust table assumes only that
every echo arrives within 16 samples, so it needs no calibration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # find the recipe book in src/

import matplotlib.pyplot as plt
import numpy as np
from ofdm_tools import (bootstrap_db, channel_model, lmmse_guess, make_dataset, nmse_db,
                        robust_table, save_results)

_, _, R_old = channel_model()                       # nominal room, learned perfectly
R_robust = robust_table(max_delay=16)               # uniform delays on [0, 16): no calibration needed
rooms = {"nominal room (8 paths)": dict(paths=8, decay=2.0),
         "changed room (12 paths)": dict(paths=12, decay=3.0)}
names = ["LS", "old-room table", "robust table", "true R of the test room"]
snrs = list(range(0, 31, 5))

lines = ["## Experiment 4: stress test, NMSE (dB), 400 test channels (seed 999)", "",
         "| test room | SNR (dB) | LS | old-room table | robust table | true R of the test room |",
         "|---|---|---|---|---|---|"]
curves = {}
for room, params in rooms.items():
    curves[room] = {name: [] for name in names}
    for snr in snrs:
        noise_var = 10 ** (-snr / 10)
        truth, obs, R_key = make_dataset(seed=999, count=400, noise_var=noise_var, **params)
        guesses = {"LS": obs,
                   "old-room table": lmmse_guess(obs, R_old, noise_var),
                   "robust table": lmmse_guess(obs, R_robust, noise_var),
                   "true R of the test room": lmmse_guess(obs, R_key, noise_var)}
        cells = []
        for name in names:
            score = nmse_db(guesses[name], truth)
            low, high = bootstrap_db(guesses[name], truth)
            curves[room][name].append(score)
            cells.append(f"{score:.1f} ± {(high - low) / 2:.1f}")
        lines.append(f"| {room} | {snr} | " + " | ".join(cells) + " |")
lines += ["", "± is half the width of the 95% bootstrap range (resampling whole test channels)."]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, room in zip(axes, rooms):
    for name, style in zip(names, ["o:", "s-", "^-", "--"]):
        color = "black" if name.startswith("true") else None
        ax.plot(snrs, curves[room][name], style, color=color, label=name)
    ax.set_title(f"Test room: {room}")
    ax.set_xlabel("SNR (dB)")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("NMSE (dB, lower is better)")
axes[1].legend()
fig.tight_layout()
save_results("exp4_mismatch", lines, fig)