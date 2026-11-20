import numpy as np
import matplotlib.pyplot as plt
from ofdm_tools import make_dataset, nmse_db, lmmse_guess, learn_table

noise_var = 0.1

# Test rooms: used ONLY for scoring, never for learning
test_truth, test_obs, R_true = make_dataset(seed=999, count=400, noise_var=noise_var)
simple_score = nmse_db(test_obs, test_truth)
key_score = nmse_db(lmmse_guess(test_obs, R_true, noise_var), test_truth)
print(f"simple guess:              {simple_score:6.1f} dB")
print(f"smart guess + answer key:  {key_score:6.1f} dB")

sizes = [8, 32, 128]                       # how many practice rooms (S)
clean_results, noisy_results = [], []
for S in sizes:
    clean_scores, noisy_scores = [], []
    for rep in range(20):                  # 20 different sets of practice rooms
        cal_truth, cal_obs, _ = make_dataset(seed=rep, count=S, noise_var=noise_var)
        R_clean = learn_table(cal_truth)              # learn from the true channels (a cheat)
        R_noisy = learn_table(cal_obs, noise_var)     # learn from noisy measurements (real life)
        clean_scores.append(nmse_db(lmmse_guess(test_obs, R_clean, noise_var), test_truth))
        noisy_scores.append(nmse_db(lmmse_guess(test_obs, R_noisy, noise_var), test_truth))
    clean_results.append(np.mean(clean_scores))
    noisy_results.append(np.mean(noisy_scores))
    print(f"S = {S:3d} practice rooms:  learned from clean {clean_results[-1]:6.1f} dB"
          f"  |  from noisy {noisy_results[-1]:6.1f} dB")

plt.plot(sizes, clean_results, "o-", label="smart guess, table learned from clean rooms")
plt.plot(sizes, noisy_results, "s-", label="smart guess, table learned from noisy rooms")
plt.axhline(simple_score, color="grey", linestyle=":", label="simple guess")
plt.axhline(key_score, color="green", linestyle="--", label="smart guess + answer key")
plt.xscale("log")
plt.xticks(sizes, [str(s) for s in sizes])
plt.xlabel("number of practice rooms S")
plt.ylabel("score (NMSE in dB, lower is better)")
plt.title("How many practice rooms does the smart guess need? (SNR 10 dB)")
plt.legend()
plt.grid(True)
plt.show()