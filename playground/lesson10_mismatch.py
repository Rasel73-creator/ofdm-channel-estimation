import numpy as np
import matplotlib.pyplot as plt
from ofdm_tools import (channel_model, make_dataset, make_other_room, nmse_db,
                        lmmse_guess, robust_table)

_, _, R_old = channel_model()            # the table for the room we practised in (8 echoes)
R_robust = robust_table(max_delay=16)    # the "pack for all weather" table

names = ["simple guess", "smart + old-room table", "smart + robust table", "smart + answer key"]

def all_scores(truth, obs, R_key, noise_var):
    return [nmse_db(obs, truth),
            nmse_db(lmmse_guess(obs, R_old, noise_var), truth),
            nmse_db(lmmse_guess(obs, R_robust, noise_var), truth),
            nmse_db(lmmse_guess(obs, R_key, noise_var), truth)]

print("columns: simple | old-room table | robust table | answer key")
for room, maker in [("SAME room (8 echoes)", make_dataset),
                    ("BIGGER room (12 echoes)", make_other_room)]:
    print(room)
    for snr in [0, 10, 20]:
        noise_var = 10 ** (-snr / 10)
        truth, obs, R_key = maker(seed=999, count=400, noise_var=noise_var)
        scores = all_scores(truth, obs, R_key, noise_var)
        print(f"  SNR {snr:2d} dB: " + "  ".join(f"{s:6.1f}" for s in scores))

# Picture: the bigger room, over many hiss levels
snr_list = list(range(0, 31, 5))
curves = []
for snr in snr_list:
    noise_var = 10 ** (-snr / 10)
    truth, obs, R_key = make_other_room(seed=999, count=400, noise_var=noise_var)
    curves.append(all_scores(truth, obs, R_key, noise_var))
curves = np.array(curves)
for i, name in enumerate(names):
    plt.plot(snr_list, curves[:, i], "o-", label=name)
plt.xlabel("SNR in dB (higher = quieter hiss)")
plt.ylabel("score (NMSE in dB, lower is better)")
plt.title("When the room changes: 12 echoes instead of 8")
plt.legend()
plt.grid(True)
plt.show()