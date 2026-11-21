"""Recipe book for the final experiments: low-rank OFDM channel estimation (Track T).

Follows the course guide and Edfors et al. (1998). Arrays hold many channels at once:
each ROW is one channel or one observation (N = 64 subcarriers by default).
"""
import sys
from pathlib import Path

import numpy as np

# The experiments save their figures and tables here (the "results" folder next to "src")
RESULTS = Path(__file__).resolve().parents[1] / "results"


# ---------- Channel model (guide, page 3) ----------
def channel_model(N=64, paths=8, decay=2.0):
    """Return F (N x L), path powers p and the covariance R = F diag(p) F^H.

    Path l arrives at delay l + 0.25 samples with power proportional to exp(-l / decay).
    The defaults (8 paths, decay 2) are the guide's model.
    """
    delays = np.arange(paths) + 0.25
    power = np.exp(-np.arange(paths) / decay)
    power /= power.sum()
    F = np.exp(-2j * np.pi * np.arange(N)[:, None] * delays[None, :] / N)
    R = (F * power[None, :]) @ F.conj().T
    return F, power, R


def draw_channel(rng, F, power):
    """One random channel h = F a, with complex Gaussian path gains of power p_l."""
    taps = np.sqrt(power / 2) * (rng.normal(size=len(power)) + 1j * rng.normal(size=len(power)))
    return F @ taps


def make_dataset(seed, count=30, noise_var=0.1, paths=8, decay=2.0):
    """Draw `count` channels h and their LS observations z = h + w (unit pilots).

    Returns (truth, observations, R_true). With the default room this is exactly the
    guide's make_dataset; paths and decay change the room for the stress test.
    """
    rng = np.random.default_rng(seed)
    F, power, R_true = channel_model(paths=paths, decay=decay)
    truth, observations = [], []
    for _ in range(count):
        h = draw_channel(rng, F, power)
        noise = np.sqrt(noise_var / 2) * (rng.normal(size=len(h)) + 1j * rng.normal(size=len(h)))
        truth.append(h)
        observations.append(h + noise)
    return np.array(truth), np.array(observations), R_true


# ---------- Scoring and uncertainty ----------
def nmse_db(guesses, truth):
    """NMSE in dB: total squared error divided by total channel energy (guide, page 3)."""
    return 10 * np.log10(np.sum(np.abs(guesses - truth) ** 2) / np.sum(np.abs(truth) ** 2))


def bootstrap_db(guesses, truth, n_boot=1000, seed=0):
    """95% range of the NMSE (dB), resampling whole channels with replacement."""
    rng = np.random.default_rng(seed)
    mistakes = np.sum(np.abs(guesses - truth) ** 2, axis=1)   # error energy of each channel
    sizes = np.sum(np.abs(truth) ** 2, axis=1)                # energy of each channel
    scores = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(truth), len(truth))
        scores.append(10 * np.log10(mistakes[pick].sum() / sizes[pick].sum()))
    return np.percentile(scores, [2.5, 97.5])


def bootstrap_mean(values, n_boot=1000, seed=0):
    """95% range of the mean of `values` (e.g. one score per calibration set), by resampling."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


# ---------- Estimators (guide, pages 1-2) ----------
def lmmse_guess(observations, R, noise_var):
    """LMMSE estimate R (R + noise_var I)^-1 z for every row z, using a linear solve (no inverse)."""
    N = R.shape[0]
    return (R @ np.linalg.solve(R + noise_var * np.eye(N), observations.T)).T


def low_rank_guess(observations, R, noise_var, rank):
    """Rank-r estimate U_r diag(lambda / (lambda + noise_var)) U_r^H z for every row z.

    Keeps the `rank` strongest eigen-directions of R and sets the others to zero.
    """
    values, vectors = np.linalg.eigh(R)
    keep = np.argsort(values)[::-1][:rank]
    values = np.maximum(values[keep], 0.0)
    U = vectors[:, keep]
    trust = values / (values + noise_var)
    return ((observations @ U.conj()) * trust) @ U.T


# ---------- Covariance tables ----------
def learn_table(examples, noise_var=0.0):
    """Estimate R from calibration rows: average of x x^H, minus noise_var * I.

    Use noise_var = 0 for clean channels (simulation privilege) and the true noise
    variance for noisy observations. Negative eigenvalues are clipped to zero.
    """
    S, N = examples.shape
    R = examples.T @ examples.conj() / S - noise_var * np.eye(N)
    values, vectors = np.linalg.eigh(R)
    return (vectors * np.maximum(values, 0.0)) @ vectors.conj().T


def shrink_table(R, alpha):
    """Shrinkage: (1 - alpha) R + alpha (trace(R) / N) I."""
    N = R.shape[0]
    return (1 - alpha) * R + alpha * (np.trace(R).real / N) * np.eye(N)


def robust_table(max_delay=16, N=64):
    """Covariance of a uniform delay profile on [0, max_delay) samples. Needs no calibration."""
    delays = np.arange(0, max_delay, 0.05)
    F = np.exp(-2j * np.pi * np.arange(N)[:, None] * delays[None, :] / N)
    return (F @ F.conj().T) / len(delays)


# ---------- Choosing settings on validation data (16 choices each: equal budgets) ----------
def tune_rank(R, val_obs, val_truth, noise_var, ranks=range(1, 17)):
    """Rank with the lowest validation NMSE. Ties go to the smaller rank."""
    scores = [nmse_db(low_rank_guess(val_obs, R, noise_var, r), val_truth) for r in ranks]
    return list(ranks)[int(np.argmin(np.round(scores, 6)))]


def tune_shrink(R, val_obs, val_truth, noise_var, alphas=np.linspace(0, 1, 16)):
    """Shrinkage alpha with the lowest validation NMSE. Ties go to the smaller alpha."""
    scores = [nmse_db(lmmse_guess(val_obs, shrink_table(R, a), noise_var), val_truth) for a in alphas]
    return float(alphas[int(np.argmin(np.round(scores, 6)))])


# ---------- Saving results ----------
def save_results(name, lines, fig=None):
    """Print the table lines, then save them to results/<name>.md and the figure to results/<name>.png."""
    RESULTS.mkdir(exist_ok=True)
    sys.stdout.reconfigure(errors="replace")   # show ? instead of crashing if the terminal lacks a symbol
    text = "\n".join(lines)
    print(text)
    (RESULTS / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    if fig is not None:
        fig.savefig(RESULTS / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved in the results folder: {name}.md" + (f" and {name}.png" if fig is not None else ""))