import numpy as np
import matplotlib.pyplot as plt
from ofdm_tools import make_dataset, nmse_db, lmmse_guess

noise_var = 0.1                                  # how loud the hiss is
snr_db = 10 * np.log10(1 / noise_var)

truth, observations, R_true = make_dataset(seed=0, count=200, noise_var=noise_var)

simple = observations                                   # simple guess: trust the measurement
smart = lmmse_guess(observations, R_true, noise_var)    # smart guess: use the moving-together table

print(f"SNR: {snr_db:.0f} dB")
print(f"simple guess score: {nmse_db(simple, truth):.1f} dB")
print(f"smart guess score:  {nmse_db(smart, truth):.1f} dB")

# Picture: one channel - truth vs both guesses
plt.plot(abs(truth[0]), linewidth=3, label="true channel")
plt.plot(abs(simple[0]), "o", markersize=4, label="simple guess")
plt.plot(abs(smart[0]), "--", linewidth=2, label="smart guess")
plt.xlabel("lane number k (0 to 63)")
plt.ylabel("arrow length (how loud each lane is)")
plt.title(f"Simple vs smart guess at SNR = {snr_db:.0f} dB")
plt.legend()
plt.grid(True)
plt.show()