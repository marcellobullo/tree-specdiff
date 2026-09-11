"""Plain-target reference sampling for the exactness figure.

Exactness here is the operational claim: every ``(rule, K, L, J)`` arm samples
the *same law as plain target sampling*. That is not the analytic mixture. The
schedule's last step is deterministic (``sigma_next = 0``), so a terminal sample
is a posterior mean and its within-component spread is 5-9% tighter than the
mixture's own sd. Every arm inherits that contraction; it has nothing to do with
speculation, so the analytic density is the wrong yardstick and a figure drawn
against it would report a bias that is not there.

Independence is the other trap. ``picard_sweep.trajectory_rngs`` keys the stream
on ``(seed, replicate)`` alone, so for one replicate every cell of a run replays
the same initial state *and* the same innovations -- a deliberate paired design
for comparing speed-ups, and fatal to a pooled distributional test: in the
canonical sweep the 324 arm samples sharing a replicate collapse to ~194
distinct states, with clusters of up to 17 identical terminal points spanning
different rules and different J. Within a *single* cell the replicates are
independent, so the law test is per arm, never pooled across cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm import models  # noqa: E402


def mixture_components(dimension: int, num_components: int, mixture_seed: int):
    """The means and sds ``MixtureFlowTarget`` draws, in its own draw order."""
    rng = np.random.default_rng(mixture_seed)
    means = rng.uniform(-2.0, 2.0, size=(num_components, dimension))
    return means, rng.uniform(0.10, 0.25, num_components)


def target_samples(setting, x0: np.ndarray, rng, chunk: int = 4096) -> np.ndarray:
    """Plain sequential target sampling: the law every arm has to match.

    The chain is the sampler's own, which is *not* the full schedule: the
    sampler is handed a standard normal, advances the ``num_steps`` stochastic
    steps, and returns; the two deterministic endpoints are priced into the cost
    baseline but never applied to the state. Running them here instead would add
    a denoising step no arm takes, contracting the within-component spread ~9%
    and reporting every arm as biased by roughly a 4% width error -- all of them
    equally, refined or not, which is the signature of a reference that is wrong
    rather than a sampler that is.
    """
    skip = set(setting.deterministic_steps)
    steps = [s for s in range(setting.total_steps) if s not in skip]
    out = np.empty_like(x0)
    for lo in range(0, len(x0), chunk):
        x = x0[lo:lo + chunk].copy()
        for step in steps:
            mean, sd = setting.target.kernel(x, step)
            x = mean + sd * rng.standard_normal(x.shape) if sd > 0.0 else mean
        out[lo:lo + chunk] = x
    return out


def assign(states: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Nearest component. Unambiguous here: the modes sit ~37 apart, radius <6."""
    return np.argmin(((states[:, None, :] - means[None]) ** 2).sum(-1), axis=1)


def whitening_frame(reference: np.ndarray, labels: np.ndarray, k: int, seed: int):
    """Per-component centre and scale from the reference, plus a fixed 2-frame.

    Whitening by component collapses all five modes onto one cloud, so the whole
    sample set is read in a single 2-D panel; the frame is a fixed random
    projection rather than a fit, so it cannot be tuned to flatter either side.
    """
    centres = np.stack([reference[labels == j].mean(0) for j in range(k)])
    scales = np.array([np.linalg.norm(reference[labels == j] - centres[j], axis=1).mean()
                       / np.sqrt(reference.shape[1]) for j in range(k)])
    frame = np.linalg.qr(np.random.default_rng(seed).standard_normal(
        (reference.shape[1], 2)))[0]
    return centres, scales, frame


def whiten(states: np.ndarray, labels: np.ndarray, centres, scales) -> np.ndarray:
    """Component-whitened deviations: unit per-coordinate scale in every mode."""
    return (states - centres[labels]) / scales[labels, None]


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample energy distance: zero exactly when the two laws coincide.

    Distances go through the Gram-matrix identity so the whole statistic is
    three BLAS calls; the naive broadcast is minutes at these sample sizes.
    """
    def mean_dist(a, b, same):
        gram = a @ b.T
        sq = (np.einsum("ij,ij->i", a, a)[:, None]
              + np.einsum("ij,ij->i", b, b)[None] - 2 * gram)
        dist = np.sqrt(np.maximum(sq, 0.0))
        if same:                       # exclude the zero diagonal
            n = len(a)
            return float((dist.sum() - np.trace(dist)) / (n * (n - 1)))
        return float(dist.mean())
    return float(2 * mean_dist(x, y, False)
                 - mean_dist(x, x, True) - mean_dist(y, y, True))
