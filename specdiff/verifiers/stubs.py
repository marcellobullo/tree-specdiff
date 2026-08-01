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


@register_verifier("rmc")
class ReflectionMaximalCoupling(Verifier):
    """Algorithm 1 (De Bortoli et al. 2025). Single proposal; chain topology."""

    max_children = 1

    def verify(self, request: VerifyRequest) -> VerifyResult:
        raise NotImplementedError(
            "Algorithm 1. Use Rank1Frame.from_request(request) for (S_hat, Z_perp, delta, e)."
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
