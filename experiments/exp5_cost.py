"""Experiment 5: computational cost. Setup (done once) vs per block (done for every OFDM block).

Setup: build the full LMMSE matrix (linear solve) or find the r = 8 strongest eigenvectors (eigh).
Per block: apply the N x N matrix, or project onto 8 directions and back (work ~ N * r).
Timed on ONE processor core, so the times show the algorithms, not how many cores the computer has.
"""
import os

for name in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[name] = "1"          # must happen before numpy is imported

import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # find the recipe book in src/

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ofdm_tools import channel_model, lmmse_guess, low_rank_guess, make_dataset, nmse_db, save_results

noise_var = 0.1
rank = 8
lanes = [64, 128, 256, 512, 1024]
blocks_per_call = 400

def fastest_time(job, tries=5):
    """Fastest of several tries, in seconds (the fastest try had the fewest interruptions)."""
    times = []
    for _ in range(tries):
        start = time.perf_counter()
        job()
        times.append(time.perf_counter() - start)
    return min(times)

def build_full(R):
    return np.linalg.solve(R + noise_var * np.eye(len(R)), R).conj().T   # A = R (R + s2 I)^-1

def build_rank(R):
    values, vectors = np.linalg.eigh(R)
    keep = np.argsort(values)[::-1][:rank]
    return vectors[:, keep], values[keep] / (values[keep] + noise_var)

rng = np.random.default_rng(0)
cost = {"setup full": [], "setup rank": [], "block full": [], "block rank": []}
for N in lanes:
    _, _, R = channel_model(N=N)
    blocks = rng.normal(size=(blocks_per_call, N)) + 1j * rng.normal(size=(blocks_per_call, N))
    cost["setup full"].append(fastest_time(lambda: build_full(R), tries=3) * 1e3)     # ms
    cost["setup rank"].append(fastest_time(lambda: build_rank(R), tries=3) * 1e3)
    A = build_full(R)
    U, trust = build_rank(R)
    cost["block full"].append(fastest_time(lambda: blocks @ A.T) / blocks_per_call * 1e6)   # us
    cost["block rank"].append(fastest_time(lambda: ((blocks @ U.conj()) * trust) @ U.T)
                              / blocks_per_call * 1e6)
    print(f"timed N = {N}")

lines = ["## Experiment 5: computational cost (one processor core)", "",
         f"Computer: {platform.processor() or platform.machine()}, Python {platform.python_version()}, "
         f"numpy {np.__version__}, matplotlib {matplotlib.__version__}", "",
         "| N | setup, full LMMSE (ms) | setup, rank-8 (ms) | per block, full LMMSE (µs) | per block, rank-8 (µs) |",
         "|---|---|---|---|---|"]
for i, N in enumerate(lanes):
    lines.append(f"| {N} | {cost['setup full'][i]:.2f} | {cost['setup rank'][i]:.2f} | "
                 f"{cost['block full'][i]:.2f} | {cost['block rank'][i]:.2f} |")
grow = lambda key: cost[key][-1] / cost[key][-3]
lines += ["", f"From N = 256 to N = 1024 (4x): setup grew {grow('setup full'):.0f}x (full) and "
          f"{grow('setup rank'):.0f}x (rank-8), rule of thumb 64x; per block grew {grow('block full'):.0f}x "
          f"(full, rule 16x) and {grow('block rank'):.1f}x (rank-8, rule 4x)."]

# Summary for the guide's list "NMSE, setup time, per-block time and rank", at N = 64 and 10 dB
truth, obs, R_true = make_dataset(seed=999, count=400, noise_var=noise_var)
lines += ["", "### Summary at N = 64, SNR 10 dB (true R, 400 test channels)", "",
          "| method | rank | NMSE (dB) | setup (ms) | per block (µs) |", "|---|---|---|---|---|",
          f"| LS | - | {nmse_db(obs, truth):.1f} | none | none |",
          f"| LMMSE | 64 (full) | {nmse_db(lmmse_guess(obs, R_true, noise_var), truth):.1f} | "
          f"{cost['setup full'][0]:.2f} | {cost['block full'][0]:.2f} |",
          f"| rank-8 | 8 | {nmse_db(low_rank_guess(obs, R_true, noise_var, rank), truth):.1f} | "
          f"{cost['setup rank'][0]:.2f} | {cost['block rank'][0]:.2f} |"]

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
left.loglog(lanes, cost["setup full"], "o-", label="full LMMSE (linear solve)")
left.loglog(lanes, cost["setup rank"], "s-", label="rank-8 (eigendecomposition)")
left.set_title("Setup: done once")
left.set_ylabel("milliseconds")
right.loglog(lanes, cost["block full"], "o-", label="full LMMSE (N x N multiply)")
right.loglog(lanes, cost["block rank"], "s-", label="rank-8 (N x 8, then 8 x N)")
right.set_title("Per block: done for every OFDM block")
right.set_ylabel("microseconds")
for ax in (left, right):
    ax.set_xticks(lanes, [str(n) for n in lanes])
    ax.set_xlabel("number of subcarriers N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
fig.tight_layout()
save_results("exp5_cost", lines, fig)