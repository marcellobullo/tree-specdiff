"""Where Algorithms 1 and 2 go.

Deliberately unimplemented. The point of these two classes is to show that the
template already carries everything the paper's rules need, and to fix their
names, topology constraints and telemetry so that filling them in is a local
edit.

Sketch of the work each one needs, in the coordinates of
:class:`~specdiff.verifiers.rank1.Rank1Frame`:

``ReflectionMaximalCoupling`` (Algorithm 1, ``K = 1``)
    Project the single child to ``s_hat``; accept with probability
    ``1 ^ phi(s_hat - delta) / phi(s_hat)``; on rejection reflect,
    ``s = delta - s_hat``; reconstruct. Acceptance probability
    ``2 * Phi_bar(-delta / 2)`` (eq. 16).

``GreedyRejectionSampling`` (Algorithm 2, any ``K``)
    Sweep the children *in the order they were drafted* -- the sequence, not
    list, coupling. Maintain the level ``lambda_k`` and residual mass
    ``G_{k+1}``; accept child ``k`` with probability
    ``1 ^ (rho(s_k) - lambda_{k-1})_+ / G_k``. On a full sweep of rejections,
    sample the normalised residual (eq. 13). The masses of the super-level
    sets are half-space masses in the projected coordinate,
    ``Q(H_k) = Phi_bar(tau_k - delta/2)`` and ``P(H_k) = Phi_bar(tau_k +
    delta/2)`` with ``tau_k = ln(lambda_k) / delta`` (Appendix B.2), so no
    numerical integration is needed.

Two traps worth writing down before either is implemented:

*   The children arrive in ``request.children`` in sampling order and a
    sequence coupling must keep it. Sorting them, or examining them by
    likelihood ratio, breaks exactness.
*   Which orthogonal residual is carried through matters. Algorithm 2 returns
    ``Z_perp,k`` on acceptance of child ``k`` but ``Z_perp,1`` on the residual
    branch (lines 9 and 18).
"""

from __future__ import annotations

from ..types import VerifyRequest, VerifyResult
from ..verify import Verifier, register_verifier
from .rank1 import Rank1Frame

import math


@register_verifier("rmc")
class ReflectionMaximalCoupling(Verifier):
    """Algorithm 1: the reflection maximal coupling of two isotropic Gaussians.

    Single proposal, so ``K = 1`` and the topology must be ``DraftTree.chain(L)``.

    In the rank-1 frame the child projects to ``S_hat ~ N(0, 1)`` under the
    proposal, while the target wants ``N(delta, 1)``. Accept with probability
    ``1 ^ phi(S_hat - delta) / phi(S_hat)``; on rejection reflect about the
    crossing point ``delta / 2`` of the two densities, ``S = delta - S_hat``,
    and carry the child's orthogonal residual through untouched. The accepted
    branch contributes ``min(p, q)`` and the reflected branch ``(q - p)_+``,
    which sum to ``q`` -- that is the exactness argument in one line.

    Per-step acceptance is ``2 * Phi_bar(delta / 2)`` (eq. 16), i.e. the overlap
    ``1 - TV(P, Q)``, which is the most any single-proposal coupling can achieve.
    """

    max_children = 1

    def verify(self, request: VerifyRequest) -> VerifyResult:

        # Backend ops
        ops = self.backend_for(request)

        ## Project the child to (s_hat, z_perp) in the rank-1 frame.
        frame = Rank1Frame.from_request(request)
        child_index = 0
        state = request.child(child_index)

        if frame.degenerate:
            # degenerate: accept anything, return the first child
            return VerifyResult(
                accepted=True,
                state=state,
                child_index=child_index,
                proposals_examined=1,
            )
        s_hat, z_perp = frame.project(state)

        # Compute the acceptance probability and accept/reject the child
        u = ops.uniform(request.rng)
        log_ratio = frame.delta*(s_hat - 0.5*frame.delta) 
        accepted = math.log1p(-u) <= log_ratio

        # On rejection, reflect the projected coordinate
        if not accepted:
            s_hat = frame.delta - s_hat
            child_index = None
            # reconstruct the new state from (s, z_perp)
            state = frame.reconstruct(s_hat, z_perp)

        return VerifyResult(
            state=state,
            accepted=accepted,
            child_index=child_index,
            proposals_examined=1,       
        )


@register_verifier("d-grs")
class GreedyRejectionSampling(Verifier):
    """Algorithm 2. Sequence coupling over the ``K`` children of a node."""

    max_children = None

    def verify(self, request: VerifyRequest) -> VerifyResult:
        raise NotImplementedError(
            "Algorithm 2. Examine request.children in the given order; "
            "return proposals_examined=k on acceptance and K + 1 on the residual branch."
        )
