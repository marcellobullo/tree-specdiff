"""Tools for validating a verification rule.

The sampler cannot establish distributional exactness from a single verifier
call. A verifier can return a Gaussian sample from the wrong distribution
without raising an error, so the library provides a reusable statistical test.

:func:`check_exactness` exploits the rank-1 reduction: under exactness the
projection of the returned state onto ``e`` is ``N(delta, 1)`` for any
``delta``, whatever the rule did internally. A one-sample KS test on that
scalar detects distributional errors without requiring SciPy.

Note that the harness projects onto the direction it *built* the synthetic node
from, not onto one recovered from the request. The two agree whenever
``delta > 0``; the distinction only matters at ``delta = 0``, where the mean
displacement vanishes and a recovered direction would be the zero vector. That
case must be supported because ``delta = 0`` is the limiting regime described
by Remark 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from .ops import default_state, resolve_backend, standard_normal_cdf
from .types import VerifyRequest


@dataclass(frozen=True)
class ExactnessReport:
    delta: float
    num_children: int
    num_samples: int
    ks_statistic: float
    critical_value: float
    acceptance_rate: float
    mean_examined: Optional[float]

    @property
    def passed(self) -> bool:
        return self.ks_statistic <= self.critical_value

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] delta={self.delta:.3f} K={self.num_children} "
            f"n={self.num_samples} KS={self.ks_statistic:.4f} "
            f"(crit {self.critical_value:.4f}) accept={self.acceptance_rate:.3f}"
        )


def check_exactness(
    verifier,
    *,
    delta: float = 1.0,
    num_children: int = 2,
    dim: int = 8,
    num_samples: int = 4000,
    alpha: float = 0.01,
    seed: int = 0,
    array_like: Any = None,
) -> ExactnessReport:
    """Run a rule ``num_samples`` times on a synthetic node and KS-test its output.

    The synthetic node represents the general case: isotropic
    Gaussians in ``dim`` dimensions whose means differ by ``delta * sigma`` in a
    random direction, matching the structure guaranteed by Equation 24 at
    every node of every tree.

    ``array_like`` sets the backend: pass ``torch.zeros(dim)`` to test a
    torch-native rule. Defaults to NumPy.

    ``seed`` controls **every** draw -- the mean direction, the children, and
    whatever the rule itself consumes, because the seeded generator is handed
    to the rule as ``request.rng``. Two calls with the same seed therefore give
    byte-identical reports. Because this is a level-``alpha`` hypothesis test,
    a correct verifier fails with probability approximately ``alpha``. Retain
    the seed when investigating failures.
    """
    if array_like is None:
        array_like = default_state(dim)
    ops = resolve_backend(array_like)
    ops.check_state_dtype(array_like, "array_like")

    rng = ops.make_rng(seed, array_like)
    direction = ops.randn_stack(1, array_like, rng)[0]
    direction = direction / ops.norm(direction)

    sigma = 0.7
    mu_p = array_like * 0.0
    mu_q = mu_p + sigma * delta * direction

    frame_projection = []
    accepted = 0
    examined = []
    verifier.reset()

    for _ in range(num_samples):
        children = ops.randn_stack(num_children, array_like, rng) * sigma + mu_p
        request = VerifyRequest(
            step=0,
            proposal_mean=mu_p,
            target_mean=mu_q,
            sigma=sigma,
            children=children,
            parent_state=mu_p,
            rng=rng,
        )
        result = verifier(request)
        # Project onto the direction this harness *constructed*, not one
        # re-derived from the request. They agree whenever delta > 0, but at
        # delta = 0 the mean displacement is the zero vector, so a frame built
        # from the request has no direction: every projection would collapse to
        # 0 and the KS statistic against N(0, 1) would be exactly 0.5 for any
        # rule, including a provably exact one. The constructed direction stays
        # a valid unit vector at every delta, and exactness still implies
        # S ~ N(delta, 1) along it -- at delta = 0 that is just N(0, 1).
        frame_projection.append(ops.dot(direction, (result.state - mu_p) / sigma))
        accepted += int(result.accepted)
        if result.proposals_examined is not None:
            examined.append(result.proposals_examined)

    stat = _ks_statistic(frame_projection, mean=delta)
    crit = math.sqrt(-0.5 * math.log(alpha / 2.0)) / math.sqrt(num_samples)
    return ExactnessReport(
        delta=delta,
        num_children=num_children,
        num_samples=num_samples,
        ks_statistic=stat,
        critical_value=crit,
        acceptance_rate=accepted / num_samples,
        mean_examined=(sum(examined) / len(examined)) if examined else None,
    )


def _ks_statistic(samples, mean: float = 0.0, std: float = 1.0) -> float:
    """Two-sided one-sample KS statistic against ``N(mean, std^2)``."""
    xs = sorted(float(x) for x in samples)
    n = len(xs)
    stat = 0.0
    for i, x in enumerate(xs):
        cdf = standard_normal_cdf((x - mean) / std)
        stat = max(stat, cdf - i / n, (i + 1) / n - cdf)
    return stat
