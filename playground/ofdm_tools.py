import numpy as np

# ---- The guide's channel code (same as Lesson 3) ----
def channel_model(N=64, paths=8):
    delays = np.arange(paths) + 0.25              # how late each echo arrives
    power = np.exp(-np.arange(paths) / 2.0)       # later echoes are weaker...
    power /= power.sum()                          # ...and together they add up to 1
    F = np.exp(-2j * np.pi * np.arange(N)[:, None] * delays[None, :] / N)
    R_true = (F * power[None, :]) @ F.conj().T    # (Lesson 5)
    return F, power, R_true

def draw_channel(rng, F, power):
    taps = np.sqrt(power / 2) * (rng.normal(size=len(power))
                                 + 1j * rng.normal(size=len(power)))
    return F @ taps                               # add up the echoes, lane by lane

# ---- The guide's test-data maker: many channels + hiss ----
def make_dataset(seed, count=30, noise_var=0.1):
    rng = np.random.default_rng(seed)
    F, power, R_true = channel_model()
    truth, observations = [], []
    for _ in range(count):
        h = draw_channel(rng, F, power)
        noise = np.sqrt(noise_var / 2) * (rng.normal(size=len(h))
                                          + 1j * rng.normal(size=len(h)))
        truth.append(h)
        observations.append(h + noise)            # pilots are all 1, so z = h + noise
    return np.array(truth), np.array(observations), R_true

# ---- The score: how big is the mistake compared with the channel? ----
def nmse_db(guesses, truth):
    mistake = np.sum(np.abs(guesses - truth) ** 2)
    size = np.sum(np.abs(truth) ** 2)
    return 10 * np.log10(mistake / size)

# ---- The smart guess (LMMSE): blend all lanes using the moving-together table ----
def lmmse_guess(observations, R, noise_var):
    N = R.shape[0]
    blended = np.linalg.solve(R + noise_var * np.eye(N), observations.T)
    return (R @ blended).T

# ---- The hidden-knobs guess (low rank): keep only the strongest knobs ----
def low_rank_guess(observations, R, noise_var, rank):
    values, vectors = np.linalg.eigh(R)              # find the knobs
    keep = np.argsort(values)[::-1][:rank]           # pick the strongest `rank` knobs
    values = np.maximum(values[keep], 0.0)
    U = vectors[:, keep]                             # their patterns
    trust = values / (values + noise_var)            # how much to trust each knob
    return np.array([U @ (trust * (U.conj().T @ z)) for z in observations])