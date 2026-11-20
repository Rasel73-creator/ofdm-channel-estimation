import numpy as np
import matplotlib.pyplot as plt
from ofdm_tools import make_dataset, nmse_db

noise_var = 0.1                               # how loud the hiss is
snr_db = 10 * np.log10(1 / noise_var)            # signal-to-hiss ratio in dB

truth, observations, R_true = make_dataset(seed=0, count=200, noise_var=noise_var)
ls_guess = observations                          # the simple guess: trust what you measured

print("channels tested:", len(truth))
print(f"SNR: {snr_db:.0f} dB")
print(f"simple guess score (NMSE): {nmse_db(ls_guess, truth):.1f} dB")

# Picture: one channel - true loudness vs guessed loudness, lane by lane
plt.plot(abs(truth[0]), linewidth=2, label="true channel |h|")
plt.plot(abs(ls_guess[0]), ".--", label="simple guess |z|")
plt.xlabel("lane number k (0 to 63)")
plt.ylabel("arrow length (how loud each lane is)")
plt.title(f"The simple guess at SNR = {snr_db:.0f} dB")
plt.legend()
plt.grid(True)
plt.show()