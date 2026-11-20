import numpy as np
import matplotlib.pyplot as plt
from ofdm_tools import make_dataset

truth, observations, R_true = make_dataset(seed=0, count=200, noise_var=0.1)

# Learn the "moving together" table from the example rooms
R_hat = np.zeros((64, 64), dtype=complex)
for h in truth:                          # for each example room...
    R_hat += np.outer(h, h.conj())       # ...multiply every lane by every (mirrored) lane
R_hat /= len(truth)                      # ...and take the average

print("average strength of lane 5:", round(abs(R_true[5, 5]), 3))
print("lanes 5 and 6 move together:", round(abs(R_true[5, 6]), 3))
print("lanes 5 and 40 move together:", round(abs(R_true[5, 40]), 3))
print("lanes 5 and 6, learned from examples:", round(abs(R_hat[5, 6]), 3))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(abs(R_true), vmin=0, vmax=1)
plt.title("answer key (R_true)")
plt.xlabel("lane j")
plt.ylabel("lane i")
plt.subplot(1, 2, 2)
plt.imshow(abs(R_hat), vmin=0, vmax=1)
plt.title(f"learned from {len(truth)} example rooms")
plt.xlabel("lane j")
plt.colorbar(label="how strongly lanes i and j move together")
plt.tight_layout()
plt.show()