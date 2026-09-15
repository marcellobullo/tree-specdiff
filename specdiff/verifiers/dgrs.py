"""Algorithm 2: diffusion greedy rejection sampling (D-GRS).

Where :mod:`specdiff.verifiers.rmc` couples one proposal maximally, D-GRS sweeps
``K`` proposals *in the order they were drafted* and takes the first that
satisfies the acceptance rule -- a *sequence* coupling, not a list coupling.

Three implementation constraints apply. The first two become relevant for
``K > 1``:

*   The children arrive in ``request.children`` in sampling order and a
    sequence coupling must keep it. Sorting them, or examining them by
    likelihood ratio, breaks exactness.
*   Acceptance keeps the accepted child's orthogonal residual. On rejection,
    ``residual_complement`` selects ``first`` (Algorithm 2's default), ``fresh``,
    or ``nearest_projection``. Selection may depend on scalar projections,
    but not on the orthogonal values.
*   Check ``frame.degenerate`` before dividing by ``delta`` to handle small,
    nonzero mean differences safely.

A note on conventions: the main text (Equations 8--11) standardizes by
``mu_p``, giving ``S ~ N(0, 1)`` under ``P``
and ``N(delta, 1)`` under ``Q``, while Appendix B.2 standardises by the
*midpoint*, giving ``N(-delta/2, 1)`` and ``N(+delta/2, 1)``. This module
follows the main text, matching :class:`~specdiff.verifiers.rank1.Rank1Frame`;
the two differ by a shift of ``delta / 2`` and the likelihood ratio is the same
function either way.
"""

from __future__ import annotations

import math

from ..ops import standard_normal_sf
from ..types import VerifyRequest, VerifyResult
from ..verify import Verifier, register_verifier
from .rank1 import Rank1Frame, RESIDUAL_COMPLEMENTS, residual_complement


def _sample_residual(frame: Rank1Frame, level: float, mass: float, u: float) -> float:
    """Draw ``S ~ r_{K+1}`` of eq. (13) by inverting its CDF.

    The residual density is ``(q(s) - level * p(s))_+ / mass`` in the projected
    coordinate. The positive part is an interval: ``rho(s) >= level`` iff
    ``s >= tau + delta / 2``, so the density is supported on ``[t, inf)`` and

        F(s) = [Q([t, s]) - level * P([t, s])] / mass

    is continuous and strictly increasing there. Inverting it by bisection needs
    only ``Phi_bar`` and requires neither SciPy nor numerical quadrature.

    This path runs only after all ``K`` proposals are rejected. Its scalar
    ``erfc`` evaluations are small relative to the target model evaluation.
    """
    delta = frame.delta
    threshold = frame.tau(level) + 0.5 * delta
    target = u * mass

    def cdf(s: float) -> float:
        q_part = standard_normal_sf(threshold - delta) - standard_normal_sf(s - delta)
        p_part = standard_normal_sf(threshold) - standard_normal_sf(s)
        return q_part - level * p_part

    # Bracket. `cdf` saturates at `mass` once `s` is far enough into the tail
    # for both survival terms to vanish, and `target < mass`, so this ends well
    # before the cap; the cap only catches `u` within rounding of 1.
    lo, hi = threshold, threshold + 1.0
    while cdf(hi) < target and hi - threshold < 64.0:
        hi = threshold + 2.0 * (hi - threshold)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, abs(lo)):
            break
    return 0.5 * (lo + hi)


@register_verifier("d-grs")
class GreedyRejectionSampling(Verifier):
    """Algorithm 2: greedy rejection sampling over the ``K`` children of a node.

    A *sequence* coupling, not a list coupling: the children are examined in the
    order they were drafted and the first one satisfying the acceptance rule is
    taken. That ordering is part of the algorithm's correctness, not a
    convenience -- sorting the children, or visiting them by likelihood ratio,
    breaks exactness.

    Round ``k`` greedily assigns as much of the remaining target mass ``G_k`` to
    child ``k`` as the coupling admits, accepting it with probability
    ``beta_k = 1 ^ (rho(S_k) - lambda_{k-1})_+ / G_k`` where
    ``rho(s) = phi(s - delta) / phi(s)`` is the target-to-proposal likelihood
    ratio. On rejection the assigned level rises to
    ``lambda_k = lambda_{k-1} + G_k``, which induces the super-level set
    ``H_k = {s : rho(s) >= lambda_k}`` and leaves
    ``G_{k+1} = Q(H_k) - lambda_k P(H_k)``. Because ``H_k`` is a half-line in the
    projected coordinate, both masses are ``Phi_bar`` evaluations
    (Appendix B.2) -- no quadrature anywhere. After ``K`` rejections the rule
    samples the normalised residual of eq. (13), which is what restores
    exactness (Theorem 1).

    Acceptance is ``1 - G_{K+1}`` (Theorem 2, eqs. 14-15). At ``K = 1`` the
    recursion gives ``lambda_1 = 1`` and ``G_2 = 2 Phi(delta/2) - 1``, so
    acceptance collapses to ``2 * Phi_bar(delta / 2)`` -- eq. (16), the same as
    RMC. The two rules differ at ``K = 1`` only in what they return on
    rejection: RMC reflects, D-GRS draws from the residual.
    """

    max_children = None

    def __init__(self, *, residual_complement: str = "first"):
        if residual_complement not in RESIDUAL_COMPLEMENTS:
            raise ValueError(f"residual_complement must be one of {RESIDUAL_COMPLEMENTS}")
        self.residual_complement = residual_complement

    def verify(self, request: VerifyRequest) -> VerifyResult:
        ops = self.backend_for(request)
        frame = Rank1Frame.from_request(request)

        # Remark 2: the kernels coincide to working precision, so rho == 1 and
        # beta_1 == 1. Short-circuiting keeps `tau` away from a zero delta.
        if frame.degenerate:
            return VerifyResult(
                state=request.child(0), accepted=True, child_index=0, proposals_examined=1
            )

        # Lines 1-3: project every child up front, in drafting order. The
        # residual branch needs Z_perp of the *first* child (line 18), so the
        # projections cannot be discarded as the sweep goes.
        projected = [frame.project(request.child(k)) for k in range(request.num_children)]

        delta = frame.delta
        half_delta = 0.5 * delta
        level = 0.0  # lambda_0
        mass = 1.0  # G_1 = Q(R) - 0 * P(R)

        for k, (s_hat, z_perp) in enumerate(projected):
            # rho enters only through min(1, (rho - level)_+ / mass), and
            # level + mass <= K + 1, so any large exponent already pins beta at
            # 1. Clamping before exp keeps a wild s_hat from raising
            # OverflowError; math.exp does not saturate to inf the way the
            # array backends do.
            rho = math.exp(min(delta * (s_hat - half_delta), 700.0))
            beta = 0.0 if mass <= 0.0 else min(1.0, max(rho - level, 0.0) / mass)

            # `<` rather than `<=`: ops.uniform draws from [0, 1), so `u == 0` is
            # attainable and `u <= beta` would accept even where beta is exactly
            # 0 -- which is the state every child below the current level is in.
            # The torch backend makes this far from academic: torch.rand is
            # float32, so P(u == 0) is 2^-24 (~6e-8), not the 2^-53 of NumPy.
            if ops.uniform(request.rng) < beta:
                return VerifyResult(
                    state=request.child(k),
                    accepted=True,
                    child_index=k,
                    proposals_examined=k + 1,
                )

            level += mass
            tau = frame.tau(level)
            # G_k is the survival probability P[reject the first k], so it is a
            # probability and it is non-increasing. Both bounds are enforced
            # rather than assumed: this is a difference of two survival
            # functions, and rounding could put it a few ulps outside either
            # end. A value above `mass` would be the damaging one -- it would
            # inflate the remaining mass and depress every later beta.
            mass = min(
                mass,
                max(
                    standard_normal_sf(tau - half_delta)
                    - level * standard_normal_sf(tau + half_delta),
                    0.0,
                ),
            )

        # The scalar residual law is independent of the perpendicular policy.
        if mass <= 0.0:
            # Unreachable except through rounding: the probability of arriving
            # here is G_{K+1} itself. Fall back to the threshold, the point the
            # residual collapses onto as its mass vanishes.
            s = frame.tau(level) + half_delta
        else:
            s = _sample_residual(frame, level, mass, ops.uniform(request.rng))

        perp = residual_complement(
            frame, request, s, self.residual_complement, projections=projected
        )
        return VerifyResult(
            state=frame.reconstruct(s, perp),
            accepted=False,
            proposals_examined=request.num_children + 1,
        )
