import time
import numpy as np
import matplotlib.pyplot as plt
from ofdm_tools import channel_model, make_dataset, nmse_db, lmmse_guess, bootstrap_db

noise_var = 0.1

# ---------- Part A: how long does each job take? ----------
def fastest_time(job, tries=5):
    times = []
    for _ in range(tries):
        start = time.perf_counter()
        job()
        times.append(time.perf_counter() - start)
    return min(times)   # the fastest try is the one where the computer wasn't busy with other things

rng = np.random.default_rng(0)
lanes = [64, 256, 1024]
setup_full, setup_knobs, block_full, block_knobs = [], [], [], []

print("Part A - how long each job takes (your computer's numbers will be different)")
for N in lanes:
    F, power, R = channel_model(N=N)
    blocks = rng.normal(size=(400, N)) + 1j * rng.normal(size=(400, N))   # 400 measurements to clean up

    # Setup jobs (done ONCE): build the full table, or find the 8 loudest knobs
    def build_full_table():
        return np.linalg.solve(R + noise_var * np.eye(N), R).conj().T
    def find_knobs():
        values, vectors = np.linalg.eigh(R)
        keep = np.argsort(values)[::-1][:8]
        return vectors[:, keep], values[keep] / (values[keep] + noise_var)
    setup_full.append(fastest_time(build_full_table) * 1e3)    # milliseconds
    setup_knobs.append(fastest_time(find_knobs) * 1e3)

    # Per-block jobs (done for EVERY measurement): use what the setup prepared
    A = build_full_table()
    U, trust = find_knobs()
    block_full.append(fastest_time(lambda: blocks @ A.T) / 400 * 1e6)                         # microseconds
    block_knobs.append(fastest_time(lambda: ((blocks @ U.conj()) * trust) @ U.T) / 400 * 1e6)

    print(f"  N = {N:4d} lanes | setup: full table {setup_full[-1]:7.1f} ms, find knobs {setup_knobs[-1]:7.1f} ms"
          f" | per block: full {block_full[-1]:6.2f} us, 8 knobs {block_knobs[-1]:5.2f} us")

print("From 256 to 1024 lanes (4 times more lanes), each job took this many times longer:")
for name, times, rule in [("setup, full table", setup_full, "4 x 4 x 4 = 64"),
                          ("setup, find knobs", setup_knobs, "4 x 4 x 4 = 64"),
                          ("per block, full", block_full, "4 x 4 = 16"),
                          ("per block, 8 knobs", block_knobs, "4")]:
    print(f"  {name:18s}: {times[2] / times[1]:5.1f} times   (rule of thumb: {rule})")

# ---------- Part B: how sure are we of each score? ----------
truth, obs, R_true = make_dataset(seed=999, count=400, noise_var=noise_var)
smart = lmmse_guess(obs, R_true, noise_var)
print(f"Part B - scores with 95% error bars ({len(truth)} test channels)")
for name, guess in [("simple guess", obs), ("smart guess", smart)]:
    low, high = bootstrap_db(guess, truth)
    print(f"  {name:12s}: {nmse_db(guess, truth):6.1f} dB   (95% range {low:6.1f} to {high:6.1f} dB)")

# ---------- Picture: two panels, because "once" and "every time" are different kinds of cost ----------
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.loglog(lanes, setup_full, "o-", label="full smart guess (build table)")
left.loglog(lanes, setup_knobs, "s-", label="8-knob guess (find knobs)")
left.set_title("Setup: done once")
left.set_ylabel("milliseconds")
right.loglog(lanes, block_full, "o-", label="full smart guess")
right.loglog(lanes, block_knobs, "s-", label="8-knob guess")
right.set_title("Per block: done for every measurement")
right.set_ylabel("microseconds")
for panel in (left, right):
    panel.set_xticks(lanes, [str(n) for n in lanes])
    panel.set_xlabel("number of lanes N")
    panel.grid(True, which="both", alpha=0.3)
    panel.legend()
plt.tight_layout()
plt.show()