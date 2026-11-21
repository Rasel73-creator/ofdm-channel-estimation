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

# ---- Learn the moving-together table from example measurements ----
def learn_table(examples, noise_var=0.0):
    S, N = examples.shape
    R = examples.T @ examples.conj() / S            # average of z z^H over the examples
    R = R - noise_var * np.eye(N)                   # take out the hiss's share (if any)
    values, vectors = np.linalg.eigh(R)             # knob strengths can't be negative,
    values = np.maximum(values, 0.0)                # so clip them at zero
    return (vectors * values) @ vectors.conj().T

# ---- Shrink a learned table towards a plain "nothing moves together" table ----
def shrink_table(R, alpha):
    N = R.shape[0]
    average = np.trace(R).real / N                  # average lane strength
    return (1 - alpha) * R + alpha * average * np.eye(N)

    # ---- A different room for stress tests: more echoes and/or slower fading ----
def make_other_room(seed, count, noise_var, paths=12, decay=3.0):
    rng = np.random.default_rng(seed)
    delays = np.arange(paths) + 0.25
    power = np.exp(-np.arange(paths) / decay)
    power /= power.sum()
    F = np.exp(-2j * np.pi * np.arange(64)[:, None] * delays[None, :] / 64)
    truth = np.array([draw_channel(rng, F, power) for _ in range(count)])
    noise = np.sqrt(noise_var / 2) * (rng.normal(size=truth.shape)
                                      + 1j * rng.normal(size=truth.shape))
    R_true = (F * power[None, :]) @ F.conj().T
    return truth, truth + noise, R_true

# ---- A robust table: assume only "every echo arrives within max_delay steps" ----
def robust_table(max_delay=16, N=64):
    delays = np.arange(0, max_delay, 0.05)           # lots of possible echo delays, evenly spread
    F = np.exp(-2j * np.pi * np.arange(N)[:, None] * delays[None, :] / N)
    return (F @ F.conj().T) / len(delays)            # every delay equally strong

# ---- Error bars: how much could a score wobble just by luck? (bootstrap) ----
def bootstrap_db(guesses, truth, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    mistakes = np.sum(np.abs(guesses - truth) ** 2, axis=1)   # mistake size, channel by channel
    sizes = np.sum(np.abs(truth) ** 2, axis=1)                # channel size, channel by channel
    scores = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(truth), len(truth))        # pull channels from a hat, with replacement
        scores.append(10 * np.log10(mistakes[pick].sum() / sizes[pick].sum()))
    return np.percentile(scores, [2.5, 97.5])                 # the middle 95% of the wobble