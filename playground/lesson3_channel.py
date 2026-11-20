import numpy as np
import matplotlib.pyplot as plt

# ---- The guide's channel code (you'll reuse it for the whole project) ----
def channel_model(N=64, paths=8):
    delays = np.arange(paths) + 0.25              # how late each echo arrives
    power = np.exp(-np.arange(paths) / 2.0)       # later echoes are weaker...
    power /= power.sum()                          # ...and together they add up to 1
    F = np.exp(-2j * np.pi * np.arange(N)[:, None] * delays[None, :] / N)
    R_true = (F * power[None, :]) @ F.conj().T    # (ignore this for now - Lesson 5)
    return F, power, R_true

def draw_channel(rng, F, power):
    taps = np.sqrt(power / 2) * (rng.normal(size=len(power))
                                 + 1j * rng.normal(size=len(power)))
    return F @ taps                               # add up the echoes, lane by lane

# ---- Let's look at it ----
F, power, R_true = channel_model()
print("echo strengths:", np.round(power, 3))
print("total strength:", round(power.sum(), 3))

rng = np.random.default_rng(0)
for i in range(3):                     # three different "rooms" (random draws)
    h = draw_channel(rng, F, power)    # 64 arrows, one per lane
    plt.plot(abs(h), marker=".", label=f"channel {i + 1}")

print("arrows in one channel:", len(h))
print("first three arrows:", np.round(h[:3], 3))

plt.xlabel("lane number k (0 to 63)")
plt.ylabel("arrow length |h_k| (how loud each lane is)")
plt.title("One channel = 64 arrows made from 8 echoes")
plt.legend()
plt.grid(True)
plt.show()