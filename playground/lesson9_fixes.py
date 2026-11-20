import numpy as np
import matplotlib.pyplot as plt
from ofdm_tools import (make_dataset, nmse_db, lmmse_guess, low_rank_guess,
                        learn_table, shrink_table)

noise_var = 0.1
ranks = range(1, 17)                    # 16 choices for fix 1 (how many knobs to keep)
alphas = np.linspace(0, 1, 16)          # 16 choices for fix 2 (how much to shrink)

# Check rooms: used to CHOOSE the settings.  Test rooms: used ONLY for the final score.
check_truth, check_obs, _ = make_dataset(seed=500, count=200, noise_var=noise_var)
test_truth, test_obs, R_true = make_dataset(seed=999, count=400, noise_var=noise_var)
simple_score = nmse_db(test_obs, test_truth)
key_score = nmse_db(lmmse_guess(test_obs, R_true, noise_var), test_truth)
print(f"simple guess:              {simple_score:6.1f} dB")
print(f"smart guess + answer key:  {key_score:6.1f} dB")

sizes = [8, 32, 128]
results = {"no fix": [], "fewer knobs": [], "shrink": []}
for S in sizes:
    no_fix, fewer, shrunk, chosen_r, chosen_a = [], [], [], [], []
    for rep in range(20):
        _, cal_obs, _ = make_dataset(seed=rep, count=S, noise_var=noise_var)   # noisy practice rooms
        R_learned = learn_table(cal_obs, noise_var)

        # Fix 1: try every number of knobs on the CHECK rooms, keep the best
        scores = [nmse_db(low_rank_guess(check_obs, R_learned, noise_var, r), check_truth)
                  for r in ranks]
        best_r = ranks[int(np.argmin(np.round(scores, 6)))]   # if tied, take the fewest knobs

        # Fix 2: try every amount of shrinking on the CHECK rooms, keep the best
        scores = [nmse_db(lmmse_guess(check_obs, shrink_table(R_learned, a), noise_var), check_truth)
                  for a in alphas]
        best_a = alphas[int(np.argmin(scores))]

        # Final exam: score everything on the TEST rooms
        no_fix.append(nmse_db(lmmse_guess(test_obs, R_learned, noise_var), test_truth))
        fewer.append(nmse_db(low_rank_guess(test_obs, R_learned, noise_var, best_r), test_truth))
        shrunk.append(nmse_db(lmmse_guess(test_obs, shrink_table(R_learned, best_a), noise_var),
                              test_truth))
        chosen_r.append(best_r)
        chosen_a.append(best_a)

    results["no fix"].append(np.mean(no_fix))
    results["fewer knobs"].append(np.mean(fewer))
    results["shrink"].append(np.mean(shrunk))
    print(f"S = {S:3d}:  no fix {np.mean(no_fix):6.1f} dB"
          f" | fewer knobs {np.mean(fewer):6.1f} dB (usually {int(np.median(chosen_r))} knobs)"
          f" | shrink {np.mean(shrunk):6.1f} dB (usually {np.median(chosen_a):.2f})")

for name, marker in [("no fix", "s-"), ("fewer knobs", "o-"), ("shrink", "^-")]:
    plt.plot(sizes, results[name], marker, label=f"smart guess, noisy table, {name}")
plt.axhline(simple_score, color="grey", linestyle=":", label="simple guess")
plt.axhline(key_score, color="black", linestyle="--", label="smart guess + answer key")
plt.xscale("log")
plt.xticks(sizes, [str(s) for s in sizes])
plt.xlabel("number of noisy practice rooms S")
plt.ylabel("score (NMSE in dB, lower is better)")
plt.title("Fixing the smart guess (SNR 10 dB)")
plt.legend()
plt.grid(True)
plt.show()