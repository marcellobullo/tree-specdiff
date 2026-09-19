"""PAWS rank selection and closed-form inverse-CDF correction.

Children are conditionally iid N(mu_p, sigma**2 I). Rank policies see only
(delta, K), never the list. The selected scalar density is
phi(s)*beta_lambda(Phi(s)). Only the optimizer may quantize delta; acceptance
and the residual CDF always use its actual value.

Two residual implementations are provided, and they sample the same law.
``inverse_cdf`` (the default) uses no numerical integration: both laws have
exact CDFs -- Phi(s - delta) for the target, and a Bernstein polynomial in
Phi(s) for the rank-selected law -- so the residual CDF is arithmetic, and the
only numerical step is locating where the two densities cross. ``rejection``
instead proposes from the target scalar N(delta, 1) and accepts with
probability (1 - p_omega/q_delta)_+, taking 1/(1 - A) trials on average but no
setup.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Sequence, Union

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.special import gammaln, log_ndtr, ndtr, ndtri
from scipy.sparse import csr_matrix, eye, hstack, vstack

from ..types import VerifyRequest, VerifyResult
from ..verify import Verifier, register_verifier
from .rank1 import Rank1Frame, RESIDUAL_COMPLEMENTS, residual_complement

RankPolicy = Union[str, Callable[[float, int], Sequence[float]]]
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SUPPORT_HALF_WIDTH = 42.0
"""Half width of the crossing scan. ``Phi(-42)`` underflows to zero, so no mass
lies outside, and a crossing missed out there contributes nothing."""
_SCAN_POINTS = 513
"""Crossing-scan resolution. Both densities are unimodal in ``Phi(s)`` up to the
rank polynomial's shape, so crossings are few and widely separated; this
resolves them with a large margin (see ``tests/test_paws_residual.py``)."""
_MIDPOINT_SERIES_LIMIT = 0.03
"""Use the midpoint series for the shift deficit while ``h*(1+|m|)`` is below
this. The first omitted term is then below ``2e-18`` relative."""


def _shift_deficit(x: float, delta: float) -> float:
    """``Phi(x) - Phi(x - delta)``, with relative accuracy at every ``delta``.

    The residual mass is a difference of shifted normal CDFs, and for small
    ``delta`` that difference loses every significant digit. As an integral it
    is symmetric about the midpoint ``m = x - delta/2``, so only even Taylor
    terms survive and the series runs in powers of ``h = delta/2`` against the
    probabilists' Hermite polynomials::

        delta * phi(m) * (1 + h^2 He_2(m)/6 + h^4 He_4(m)/120 + h^6 He_6(m)/5040)

    Outside its truncation range ``delta`` is large enough that differencing the
    CDFs keeps full accuracy, provided each term is taken in the smaller tail.
    Infinite endpoints give exactly zero, which is what the mass formula needs.
    """
    if not math.isfinite(x):
        return 0.0
    h = 0.5 * delta
    m = x - h
    if h * (1.0 + abs(m)) < _MIDPOINT_SERIES_LIMIT:
        square, h2 = m * m, h * h
        he2 = square - 1.0
        he4 = square * (square - 6.0) + 3.0
        he6 = square * (square * (square - 15.0) + 45.0) - 15.0
        series = 1.0 + h2 * (he2 / 6.0 + h2 * (he4 / 120.0 + h2 * he6 / 5040.0))
        return delta * math.exp(-0.5 * m * m - _LOG_SQRT_2PI) * series
    if x - delta >= 0.0:
        return float(ndtr(delta - x) - ndtr(-x))
    return float(ndtr(x) - ndtr(x - delta))


def _weights(values: Sequence[float], k: int) -> tuple[float, ...]:
    values = tuple(float(v) for v in values)
    if len(values) != k or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("rank policy must return K finite nonnegative weights")
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("rank weights must have positive finite sum")
    return tuple(v / total for v in values)


def _log_basis(s, k):
    """Log Beta order-statistic densities at Phi(s), without forming Phi(s)."""
    s = np.asarray(s, dtype=np.float64)
    r = np.arange(k, dtype=np.float64)
    return (
        gammaln(k + 1) - gammaln(r + 1) - gammaln(k - r)
        + log_ndtr(s)[..., None] * r
        + log_ndtr(-s)[..., None] * (k - 1 - r)
    )


@lru_cache(maxsize=512)
def _optimized_weights(delta: float, k: int) -> tuple[float, ...]:
    uniform = (1.0 / k,) * k
    maximum = (0.0,) * (k - 1) + (1.0,)
    if k == 1 or delta < 1e-7:
        return uniform
    # Beyond this range finite grids resolve negligible overlap. This policy
    # fallback changes efficiency only; correction still uses the exact gap.
    if delta > 12:
        return maximum
    s = np.unique(np.r_[np.linspace(-9, 9, 361),
                        np.linspace(delta - 9, delta + 9, 361)])
    widths = np.diff(s)
    integration_weights = np.r_[widths[0], widths[:-1] + widths[1:], widths[-1]] / 2
    basis = np.exp(_log_basis(s, k) - 0.5 * s[:, None] ** 2 - _LOG_SQRT_2PI)
    q = np.exp(-0.5 * (s - delta) ** 2 - _LOG_SQRT_2PI)
    n = len(s)
    identity = eye(n, format="csr")
    constraints = vstack([
        hstack([csr_matrix((n, k)), identity]),
        hstack([-csr_matrix(basis), identity]),
    ], format="csr")
    out = linprog(
        np.r_[np.zeros(k), -integration_weights],
        A_ub=constraints, b_ub=np.r_[q, np.zeros(n)],
        A_eq=csr_matrix(np.r_[np.ones(k), np.zeros(n)][None]), b_eq=[1.0],
        bounds=(0.0, None), method="highs-ds",
        options={"primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9},
    )
    candidates = [uniform, maximum]
    if out.success:
        candidates.append(_weights(np.maximum(out.x[:k], 0.0), k))
    return max(candidates, key=lambda w: float(
        integration_weights @ np.minimum(q, basis @ w)
    ))


RESIDUAL_METHODS = ("inverse_cdf", "rejection")
"""How the correction draws from ``(q_delta - p_omega)_+``; see
:func:`_sample_residual_by_rejection` and :class:`_ResidualCDF`."""

DEFAULT_REJECTION_TRIAL_CAP = 10_000
"""Trials after which the rejection loop defers to the closed form.

The loop is geometric with success probability ``1 - A``, so the cap is reached
with probability ``A ** cap`` -- negligible except where acceptance is so close
to one that the residual is almost never needed. The fallback draw is an
independent exact residual sample, and the discarded trials carry no
information about it, so the capped sampler still has exactly the residual law:
the cap bounds work, it does not approximate.
"""


def rank_weights(delta: float, k: int, policy: RankPolicy = "optimized") -> tuple[float, ...]:
    """List-independent weights; small gaps retain relative resolution."""
    if k < 1 or not math.isfinite(delta) or delta < 0:
        raise ValueError("rank weights require finite delta >= 0 and K >= 1")
    if callable(policy):
        return _weights(policy(delta, k), k)
    if policy == "uniform":
        return (1.0 / k,) * k
    if policy == "max":
        return (0.0,) * (k - 1) + (1.0,)
    if policy != "optimized":
        raise ValueError("rank_policy must be 'optimized', 'uniform', 'max', or callable")
    return _optimized_weights(float(f"{delta:.3g}"), k)


class _SelectedDensity:
    """``log beta_lambda(Phi(s))``, in whichever form keeps relative accuracy.

    The optimizer's weights are often uniform perturbed at order ``delta``, and
    then ``log beta`` is itself ``O(delta)``. A log-sum-exp over the rank terms
    returns that with ``O(1e-16)`` *absolute* error, which is ``1e-16/delta``
    relative, and acceptance and the crossing locations both inherit it. The
    uniform combination of Beta order-statistic densities is exactly ``1``, so
    subtracting it first and using ``log1p`` keeps the correction relative
    instead. Concentrated weights are the opposite case -- there ``beta`` itself
    underflows in the tails and the log-sum-exp is the accurate form -- so each
    is used where it holds. :meth:`cdf_deviation` applies the same idea one
    integral up, where it is what keeps the residual mass relative.
    """

    def __init__(self, weights):
        self.weights = weights
        self.uniform = all(w == weights[0] for w in weights)
        k = len(weights)
        self.terms = tuple(
            (r, k - 1 - r, math.log(w) + math.lgamma(k + 1)
             - math.lgamma(r + 1) - math.lgamma(k - r))
            for r, w in enumerate(weights) if w > 0
        )
        # Signed deviations from uniform, with the log-basis coefficient folded
        # in. Their weighted sum is `beta - 1`.
        self.deviations = tuple(
            (r, k - 1 - r, w - 1.0 / k, math.lgamma(k + 1)
             - math.lgamma(r + 1) - math.lgamma(k - r))
            for r, w in enumerate(weights) if w != 1.0 / k
        )
        # The same deviation idea one integral up, for `cdf_deviation`. The
        # cumulative weights `Lambda_j` are the Bernstein coefficients of the
        # selected CDF, and `j / k` are those of `Phi` itself.
        cumulative = tuple(math.fsum(weights[:j]) for j in range(1, k + 1))
        self.cdf_deviations = tuple(
            (j, cumulative[j - 1] - j / k, math.lgamma(k + 1)
             - math.lgamma(j + 1) - math.lgamma(k - j + 1))
            for j in range(1, k + 1) if cumulative[j - 1] != j / k
        )
        self._rows = np.array([[a, b, c] for a, b, c in self.terms], dtype=np.float64)

    def log_beta(self, s):
        if self.uniform:
            return 0.0
        left, right = float(log_ndtr(s)), float(log_ndtr(-s))
        correction = math.fsum(
            dev * math.exp(c + a * left + b * right)
            for a, b, dev, c in self.deviations
        )
        # log1p is the accurate branch only while `beta` stays near 1; further
        # out the terms themselves carry the magnitude and log-sum-exp is exact.
        if -0.5 < correction < 0.5:
            return math.log1p(correction)
        terms = [c + a * left + b * right for a, b, c in self.terms]
        top = max(terms)
        return top + math.log(math.fsum(math.exp(t - top) for t in terms))

    def log_beta_many(self, s):
        """``log_beta`` over an array, for the crossing scan.

        Log-sum-exp only: the scan needs signs on a coarse grid, and every
        candidate it reports is refined with the scalar form above.
        """
        s = np.asarray(s, dtype=np.float64)
        if self.uniform:
            return np.zeros(s.shape, dtype=np.float64)
        a, b, c = self._rows.T
        terms = c + np.outer(log_ndtr(s), a) + np.outer(log_ndtr(-s), b)
        top = terms.max(axis=1, keepdims=True)
        return (top + np.log(np.exp(terms - top).sum(axis=1, keepdims=True)))[:, 0]

    def cdf_deviation(self, s):
        """``P(S_selected <= s) - Phi(s)``, in closed form.

        ``beta_lambda`` is a degree ``K-1`` polynomial in ``u = Phi(s)``, being a
        mixture of Beta order-statistic densities with integer parameters. So its
        integral is the degree ``K`` Bernstein polynomial whose coefficients are
        the cumulative rank weights::

            P(S_selected <= s) = sum_j Lambda_j C(K,j) u^j (1-u)^(K-j)

        Subtracting ``Phi`` costs nothing and buys everything, because
        ``sum_j (j/K) C(K,j) u^j (1-u)^(K-j)`` is exactly ``u``: the coefficients
        become ``Lambda_j - j/K``, which vanish identically for uniform weights
        and otherwise stay the size of the deviation rather than the size of the
        CDF. That is what keeps the residual mass relative at small ``delta``.
        Both infinite endpoints give exactly zero, since ``Lambda_K = 1``.
        """
        if self.uniform or not math.isfinite(s):
            return 0.0
        k = len(self.weights)
        left, right = float(log_ndtr(s)), float(log_ndtr(-s))
        return math.fsum(
            dev * math.exp(c + j * left + (k - j) * right)
            for j, dev, c in self.cdf_deviations
        )


@lru_cache(maxsize=512)
def _selected_density(weights: tuple[float, ...]) -> _SelectedDensity:
    """Shared, immutable ``_SelectedDensity`` for a weight vector.

    Acceptance builds one per node and the rejection loop queries one per trial,
    so the Bernstein bookkeeping is worth keeping rather than rebuilding.
    """
    return _SelectedDensity(weights)


class _ResidualCDF:
    """Inverse CDF of the entire positive-part residual, in closed form.

    ``f = N(delta, 1)`` and ``g(s) = phi(s) beta_lambda(Phi(s))`` both have exact
    CDFs, so the residual mass on an interval where ``f > g`` is arithmetic::

        mass(a, b) = (T(a) - T(b)) - (D(b) - D(a))

    with ``T`` the shift deficit ``Phi(x) - Phi(x - delta)`` and ``D`` the
    selected CDF's deviation from ``Phi``. Both vanish at ``+-infinity``, so the
    two unbounded spans need no special case, and both are the size of the
    residual rather than the size of a CDF, so nothing cancels catastrophically.

    The only numerical step is locating the crossings of ``f - g``, and that step
    is well conditioned in exactly the way quadrature was not: ``f - g`` vanishes
    *linearly* at a crossing, so an error ``eps`` in a crossing location costs
    only ``O(eps^2)`` of mass. Support may be disconnected -- two spans is the
    normal case, and maximum-rank selection puts one in each tail -- so spans are
    found, not assumed.
    """

    def __init__(self, delta, weights):
        self.delta = delta
        self.selected = _selected_density(weights)
        self.bounds = (min(0.0, delta) - _SUPPORT_HALF_WIDTH,
                       max(0.0, delta) + _SUPPORT_HALF_WIDTH)
        self.spans = self._spans()
        self.masses = tuple(self.mass(a, b) for a, b in self.spans)
        self.total = math.fsum(self.masses)
        if not math.isfinite(self.total) or self.total <= 0:
            raise FloatingPointError(
                "PAWS residual is empty or below numerical resolution: "
                f"delta={self.delta!r}, K={len(weights)}"
            )

    def log_ratio(self, z):
        """Log selected/target density at ``s = z + delta``; zeros bound support."""
        s = z + self.delta
        return self.selected.log_beta(s) - self.delta * (z + 0.5 * self.delta)

    def pdf(self, z):
        """Residual density at ``s = z + delta``. Diagnostic: the mass formula
        is closed form and never evaluates it."""
        log_ratio = self.log_ratio(z)
        return math.exp(-0.5 * z * z - _LOG_SQRT_2PI) * (-math.expm1(min(0.0, log_ratio)))

    def log_excess(self, s):
        """``log f - log g`` at ``s``. Positive exactly on the residual support."""
        return -self.log_ratio(s - self.delta)

    def _spans(self):
        """The intervals where ``f > g``, as a tuple of ``(a, b)`` pairs.

        A vectorised sweep proposes the cells that change sign and the scalar
        form refines each one, so a crossing is never placed by the coarse grid.
        Outside ``bounds`` both densities underflow, so the classification of the
        two unbounded spans is decided at the scan edge: if a crossing hides
        beyond it, both that span's mass and its misclassification are zero.
        """
        lo, hi = self.bounds
        grid = np.linspace(lo, hi, _SCAN_POINTS)
        excess = (self.delta * (grid - 0.5 * self.delta)
                  - self.selected.log_beta_many(grid)) > 0
        cuts = []
        for i in np.nonzero(excess[:-1] != excess[1:])[0]:
            left, right = float(grid[i]), float(grid[i + 1])
            if self.log_excess(left) * self.log_excess(right) < 0:
                cuts.append(brentq(self.log_excess, left, right,
                                   xtol=1e-15, rtol=8.9e-16))
            else:
                # The scalar and vectorised forms disagree only where the excess
                # is zero to rounding; the cell boundary is then as good a root.
                cuts.append(left if abs(self.log_excess(left))
                            <= abs(self.log_excess(right)) else right)
        edges = (-math.inf, *cuts, math.inf)
        spans = []
        for a, b in zip(edges[:-1], edges[1:]):
            probe = (lo if a == -math.inf else
                     hi if b == math.inf else 0.5 * (a + b))
            if self.log_excess(probe) > 0:
                spans.append((a, b))
        return tuple(spans)

    def mass(self, a, b):
        """Residual mass on ``(a, b)``, which must lie within a single span."""
        return ((_shift_deficit(a, self.delta) - _shift_deficit(b, self.delta))
                - (self.selected.cdf_deviation(b) - self.selected.cdf_deviation(a)))

    def sample(self, u):
        # An attainable zero or one uniform must not produce an infinite state.
        u = min(max(float(u), np.finfo(float).eps / 2), 1 - np.finfo(float).eps / 2)
        wanted, accumulated = u * self.total, 0.0
        span, mass = self.spans[-1], self.masses[-1]
        for (a, b), m in zip(self.spans, self.masses):
            if wanted <= accumulated + m:
                span, mass = (a, b), m
                break
            accumulated += m
        else:
            accumulated = math.fsum(self.masses[:-1])
        a, b = span
        lo = a if math.isfinite(a) else self.bounds[0]
        hi = b if math.isfinite(b) else self.bounds[1]
        target = min(max(wanted - accumulated, 0.0), mass)
        # Clamping the bracket on the achieved mass, not on `mass`, keeps the
        # sign change inside it even when the truncated tail is not exactly zero.
        if target <= 0.0 or self.mass(a, lo) >= target:
            return lo
        if self.mass(a, hi) <= target:
            return hi
        return brentq(lambda s: self.mass(a, s) - target, lo, hi,
                      xtol=2e-13, rtol=1e-14)


@lru_cache(maxsize=128)
def _residual_cdf(delta, weights):
    # Exact delta key: optimizer rounding must never leak into correction.
    return _ResidualCDF(delta, weights)


def _sample_residual_by_rejection(delta, weights, ops, rng, max_trials):
    """Draw from ``(q_delta - p_omega)_+`` by rejection, as ``(s, trials)``.

    Propose ``S ~ N(delta, 1)``, the target scalar, and keep it with probability
    ``(1 - p_omega(S)/q_delta(S))_+``. The kept density is then
    ``q_delta * (1 - p_omega/q_delta)_+ = (q_delta - p_omega)_+`` up to its
    normalization, which is exactly the residual. No target-model evaluation is
    involved, and unlike :class:`_ResidualCDF` there is no support geometry to
    work out -- disconnected support costs nothing here, since a proposal
    landing where ``p_omega >= q_delta`` simply has acceptance probability zero.

    The price is the trial count: acceptance per trial is the residual mass
    ``1 - A``, so the loop runs ``1/(1 - A)`` times on average *given* that a
    correction is needed. Proposals come from the same uniform stream as every
    other decision, through ``ndtri``, so the sampler stays reproducible.
    """
    selected = _selected_density(weights)
    half_eps = 0.5 * np.finfo(float).eps
    for trial in range(1, max_trials + 1):
        # An attainable zero or one uniform must not produce an infinite state.
        u = min(max(float(ops.uniform(rng)), half_eps), 1.0 - half_eps)
        s = delta + float(ndtri(u))
        # log p_omega(s) - log q_delta(s), the form `_ResidualCDF.log_ratio` uses.
        log_ratio = selected.log_beta(s) - delta * (s - 0.5 * delta)
        if ops.uniform(rng) < -math.expm1(min(0.0, log_ratio)):
            return s, trial
    return _residual_cdf(delta, weights).sample(ops.uniform(rng)), max_trials


@register_verifier("paws")
class RankSelectionCoupling(Verifier):
    """PAWS with rank-selected acceptance and first-child complement reuse.

    rank_policy: 'optimized', 'uniform', 'max', or callable (delta, K) -> weights.
    residual_complement: 'first' (D-GRS default), 'fresh', 'nearest_projection'.
    residual_method: 'inverse_cdf' (default) or 'rejection'. Both sample the
    same residual law; they differ in cost, not in distribution.
    Policies preserve the target law up to floating-point tolerances; the
    residual CDF is closed form, so no integration tolerance enters, and the
    rejection loop introduces none of its own either.
    No temperature or biased acceptance mode is provided.

    :attr:`residual_draws` and :attr:`residual_trials` count corrections and the
    proposals the rejection loop spent on them since the last :meth:`reset`, so
    a run can report the realized trials per correction. The inverse-CDF method
    spends no trials and leaves the second counter at zero.
    """

    def __init__(self, *, rank_policy: RankPolicy = "optimized",
                 residual_complement: str = "first",
                 residual_method: str = "inverse_cdf",
                 residual_max_trials: int = DEFAULT_REJECTION_TRIAL_CAP):
        if not callable(rank_policy) and rank_policy not in ("optimized", "uniform", "max"):
            raise ValueError("invalid rank_policy")
        if residual_complement not in RESIDUAL_COMPLEMENTS:
            raise ValueError("invalid residual_complement")
        if residual_method not in RESIDUAL_METHODS:
            raise ValueError(f"residual_method must be one of {RESIDUAL_METHODS}")
        if int(residual_max_trials) != residual_max_trials or residual_max_trials < 1:
            raise ValueError("residual_max_trials must be a positive integer")
        self.rank_policy = rank_policy
        self.residual_complement = residual_complement
        self.residual_method = residual_method
        self.residual_max_trials = int(residual_max_trials)
        self.residual_draws = 0
        self.residual_trials = 0

    def reset(self) -> None:
        self.residual_draws = 0
        self.residual_trials = 0

    def _sample_residual(self, delta, weights, ops, rng) -> float:
        """The corrected scalar, by whichever residual implementation is set."""
        if self.residual_method == "rejection":
            s, trials = _sample_residual_by_rejection(
                delta, weights, ops, rng, self.residual_max_trials)
        else:
            s, trials = _residual_cdf(delta, weights).sample(ops.uniform(rng)), 0
        self.residual_draws += 1
        self.residual_trials += trials
        return s

    def verify(self, request: VerifyRequest) -> VerifyResult:
        if not math.isfinite(request.sigma) or request.sigma <= 0 or request.num_children < 1:
            raise ValueError("PAWS requires sigma > 0 and at least one child")
        ops = self.backend_for(request)
        frame = Rank1Frame.from_request(request)
        if not math.isfinite(frame.delta):
            raise ValueError("PAWS requires a finite normalized mean gap")
        if frame.degenerate:
            return VerifyResult(request.child(0), True, 0, proposals_examined=1)
        k = request.num_children
        weights = rank_weights(frame.delta, k, self.rank_policy)
        scores = [ops.dot(frame.direction, (request.child(j) - frame.mu_p) / frame.sigma)
                  for j in range(k)]
        order = sorted(range(k), key=scores.__getitem__)
        draw, cumulative = ops.uniform(request.rng), 0.0
        rank = max(r for r, weight in enumerate(weights) if weight > 0)
        for r, weight in enumerate(weights):
            cumulative += weight
            if draw < cumulative:
                rank = r
                break
        child = order[rank]
        log_accept = min(0.0, frame.delta * (scores[child] - 0.5 * frame.delta)
                         - _selected_density(weights).log_beta(scores[child]))
        u = ops.uniform(request.rng)
        if (math.log(u) if u > 0 else -math.inf) < log_accept:
            return VerifyResult(request.child(child), True, child, proposals_examined=k)
        s = self._sample_residual(frame.delta, weights, ops, request.rng)
        perp = residual_complement(frame, request, s, self.residual_complement)
        return VerifyResult(frame.reconstruct(s, perp), False, proposals_examined=k)
