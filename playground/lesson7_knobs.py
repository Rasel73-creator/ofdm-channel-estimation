import numpy as np
import matplotlib.pyplot as plt
from ofdm_tools import channel_model, make_dataset, nmse_db, lmmse_guess, low_rank_guess

noise_var = 0.1
truth, observations, R_true = make_dataset(seed=0, count=200, noise_var=noise_var)
F, power, _ = channel_model()

# 1. Find the knobs and their strengths (biggest first)
strengths = np.linalg.eigvalsh(R_true)[::-1]
print("strongest 10 knobs:", np.round(strengths[:10], 2))
print("echo strengths x 64:", np.round(64 * power, 2))

# 2. Guess using only the strongest few knobs
smart = lmmse_guess(observations, R_true, noise_var)
print(f"smart guess (Lesson 6): {nmse_db(smart, truth):.1f} dB")
ranks = range(1, 17)
scores = []
for rank in ranks:
    guess = low_rank_guess(observations, R_true, noise_var, rank)
    scores.append(nmse_db(guess, truth))
for rank in [1, 2, 4, 8, 16]:
    print(f"keep {rank:2d} knobs: {scores[rank - 1]:.1f} dB")

# 3. Pictures
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(64), strengths)
plt.xlabel("knob number (strongest first)")
plt.ylabel("knob strength")
plt.title("How strong is each hidden knob?")
plt.subplot(1, 2, 2)
plt.plot(list(ranks), scores, "o-")
plt.axhline(nmse_db(observations, truth), color="grey", linestyle=":", label="simple guess")
plt.xlabel("number of knobs kept")
plt.ylabel("score (NMSE in dB, lower is better)")
plt.title("Score vs number of knobs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()