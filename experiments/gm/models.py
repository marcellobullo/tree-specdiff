"""The Gaussian-mixture target of the paper's Section 5.1, for the sweeps.

A faithful port of the reference implementation's model + scheduler pair:

* the analytic posterior-mean velocity of an isotropic Gaussian mixture under
  the linear flow-matching interpolant ``x_sigma = (1 - sigma) x_0 + sigma xi``;
* the churn reverse-SDE kernel, on the uniform ``sigma`` grid that
  ``diffusers.FlowMatchEulerDiscreteScheduler`` produces at its defaults
  (``shift = 1``, ``num_train_timesteps = 1000``).

Verified against the reference implementation on identical inputs: sigma grid
to 3e-8, velocity to 2e-7, kernel mean to 4e-8, kernel std exactly.

Conventions worth knowing before reading the numbers
----------------------------------------------------
``sigma`` runs from 1 (pure noise) down to 0 (data), so it is the *interpolant
coefficient*, not the transition's noise scale. The transition std is
``eps * g(sigma) * sqrt(-dt)``, which is what :class:`ChurnSchedule` returns.

Two of the ``T`` steps are **deterministic** and carry zero transition noise:
the first (``sigma = 1``, where ``g`` diverges) and the last (``sigma_next =
0``). Speculation is vacuous at zero noise -- proposal and target are distinct
point masses, so nothing can be coupled -- and ``NoiseSchedule`` refuses a
non-positive scale outright (Remark 3). :func:`build` therefore exposes the
``T - 2`` stochastic steps and reports the two skipped ones, rather than
silently reindexing: a run of ``T = 30`` here is 28 speculative steps plus 2
steps every sampler pays deterministically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from specdiff import NoiseSchedule, TargetTransition

NUM_TRAIN_TIMESTEPS = 1000


def sigma_grid(num_steps: int, shift: float = 1.0) -> np.ndarray:
    """The ``(num_steps + 1,)`` sigma grid, terminal 0 included."""
    timesteps = np.linspace(NUM_TRAIN_TIMESTEPS, 1.0, num_steps)
    sigmas = timesteps / NUM_TRAIN_TIMESTEPS
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return np.concatenate([sigmas, np.zeros(1)])


class MixtureFlowTarget(TargetTransition):
    """``m^q`` for the mixture: one churn reverse step, exact velocity."""

    def __init__(
        self,
        dimension: int,
        num_components: int,
        sigmas: np.ndarray,
        eps: float,
        *,
        sd_low: float = 0.10,
        sd_high: float = 0.25,
        mixture_seed: int = 20260714,
        s_noise: float = 1.0,
        step_offset: int = 0,
    ) -> None:
        super().__init__()
        # Draw order matters: it is what makes this the *same* mixture as the
        # reference implementation for a given seed.
        rng = np.random.default_rng(mixture_seed)
        self.means_ = rng.uniform(-2.0, 2.0, size=(num_components, dimension))
        self.sds = rng.uniform(sd_low, sd_high, num_components)
        self.dimension = dimension
        self.sigmas = sigmas
        self.eps = float(eps)
        self.s_noise = float(s_noise)
        self.step_offset = int(step_offset)

    def velocity(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """``v = (x - E[x_0 | x]) / sigma``, the posterior-mean velocity."""
        a = 1.0 - sigma
        var = a**2 * self.sds**2 + sigma**2
        diff = x[:, None, :] - a * self.means_[None]
        logw = -0.5 * (diff**2).sum(-1) / var - 0.5 * self.dimension * np.log(var)
        logw -= logw.max(axis=1, keepdims=True)
        resp = np.exp(logw)
        resp /= resp.sum(axis=1, keepdims=True)
        gain = (a * self.sds**2 / var)[None, :, None]
        x0_hat = (resp[:, :, None] * (self.means_[None] + gain * diff)).sum(axis=1)
        return (x - x0_hat) / sigma

    def kernel(self, x: np.ndarray, step: int) -> Tuple[np.ndarray, float]:
        """``(mean, std)`` of one reverse step, in *absolute* step indices."""
        sigma, sigma_next = self.sigmas[step], self.sigmas[step + 1]
        dt = sigma_next - sigma  # < 0
        v = self.velocity(x, sigma)
        stochastic = (
            self.eps > 0.0
            and 1e-6 < sigma < 1.0 - 1e-6
            and sigma_next > 0.0
        )
        if not stochastic:
            return x + dt * v, 0.0  # deterministic Euler fallback
        g2 = 2.0 * sigma / (1.0 - sigma)
        score = -(x + (1.0 - sigma) * v) / sigma
        drift = v - 0.5 * self.eps**2 * g2 * score
        std = self.eps * math.sqrt(g2) * math.sqrt(-dt) * self.s_noise
        return x + dt * drift, std

    def means(self, indices_in_batch, states, steps):
        out = np.empty_like(states)
        for step in sorted(set(steps)):
            idx = [i for i, s in enumerate(steps) if s == step]
            out[idx] = self.kernel(states[idx], step + self.step_offset)[0]
        return out

    def affine(self, step: int) -> Tuple[float, float]:
        """``(a, b)`` with ``mean = a x + b v``, in *absolute* step indices.

        :meth:`kernel` written out: with ``c = (1/2) eps^2 g^2 / sigma`` the
        churn mean is ``x (1 + dt c) + v dt (1 + c (1 - sigma))``; the
        deterministic fallback is ``x + dt v``.
        """
        sigma, sigma_next = self.sigmas[step], self.sigmas[step + 1]
        dt = sigma_next - sigma  # < 0
        stochastic = (
            self.eps > 0.0
            and 1e-6 < sigma < 1.0 - 1e-6
            and sigma_next > 0.0
        )
        if not stochastic:
            return 1.0, float(dt)
        g2 = 2.0 * sigma / (1.0 - sigma)
        c = 0.5 * self.eps**2 * g2 / sigma
        return float(1.0 + dt * c), float(dt * (1.0 + c * (1.0 - sigma)))

    def freeze_drift(self, states, means, steps):
        """The velocity behind ``means``: ``v = (m - a x) / b``.

        Freezing ``v`` rather than ``m - x`` makes the delayed-drift proposal
        re-evaluate the churn score correction at the drafted node's own state
        and step; see :meth:`TargetTransition.freeze_drift`.
        """
        out = np.empty_like(states)
        for step in sorted(set(steps)):
            idx = [i for i, s in enumerate(steps) if s == step]
            a, b = self.affine(step + self.step_offset)
            out[idx] = (means[idx] - a * states[idx]) / b
        return out

    def apply_drift(self, drift, states, steps):
        """One churn step at ``(states, steps)`` with the frozen velocity."""
        out = np.empty_like(states)
        for step in sorted(set(steps)):
            idx = [i for i, s in enumerate(steps) if s == step]
            a, b = self.affine(step + self.step_offset)
            out[idx] = a * states[idx] + b * drift[idx]
        return out


class ChurnSchedule(NoiseSchedule):
    """``sigma_n`` of the churn kernel. Cached: it is state-independent."""

    def __init__(self, target: MixtureFlowTarget) -> None:
        self.target = target
        self._cache: dict = {}

    def sigma(self, step: int) -> float:
        if step not in self._cache:
            probe = np.zeros((1, self.target.dimension))
            absolute = step + self.target.step_offset
            self._cache[step] = float(self.target.kernel(probe, absolute)[1])
        return self._cache[step]


@dataclass(frozen=True)
class Setting:
    """One fully-specified experimental configuration."""

    target: MixtureFlowTarget
    schedule: ChurnSchedule
    dimension: int
    num_steps: int
    """Speculative steps -- ``T`` minus the deterministic endpoints."""
    total_steps: int
    """``T`` as passed; the cost baseline a standard sampler pays."""
    deterministic_steps: Tuple[int, ...]

    def initial_state(self, rng) -> np.ndarray:
        return rng.standard_normal(self.dimension)


def build(
    dimension: int = 512,
    num_components: int = 5,
    num_steps: int = 30,
    eps: float = 0.06,
    *,
    mixture_seed: int = 20260714,
    sd_low: float = 0.10,
    sd_high: float = 0.25,
) -> Setting:
    """Build the setting, skipping the deterministic endpoints (see module doc)."""
    sigmas = sigma_grid(num_steps)
    probe = MixtureFlowTarget(dimension, num_components, sigmas, eps,
                              sd_low=sd_low, sd_high=sd_high, mixture_seed=mixture_seed)
    zero = tuple(s for s in range(num_steps) if probe.kernel(np.zeros((1, dimension)), s)[1] == 0.0)
    leading = 0
    while leading in zero:
        leading += 1
    trailing = 0
    while (num_steps - 1 - trailing) in zero:
        trailing += 1

    target = MixtureFlowTarget(dimension, num_components, sigmas, eps, sd_low=sd_low,
                               sd_high=sd_high, mixture_seed=mixture_seed,
                               step_offset=leading)
    return Setting(
        target=target,
        schedule=ChurnSchedule(target),
        dimension=dimension,
        num_steps=num_steps - leading - trailing,
        total_steps=num_steps,
        deterministic_steps=zero,
    )
