"""Algorithm 1: the reflection maximal coupling (De Bortoli et al. 2025).

The single-proposal rule. It works entirely in the coordinates of
:class:`~specdiff.verifiers.rank1.Rank1Frame`, where the ``d``-dimensional
coupling collapses to a scalar one and the orthogonal residual of the proposal
rides through untouched.

Two numerical points that generalise to any rule written against this contract:

*   Never form the Gaussian density. The acceptance ratio simplifies to a single
    exponential -- the quadratics cancel, the normalising constant cancels, and
    you avoid differencing two large nearly equal squared norms. This is why
    :mod:`specdiff.ops` ships ``Phi`` and ``Phi_bar`` and no PDF.
*   Compare in log space with ``math.log1p(-u)``, not ``math.log(u)``.
    ``ops.uniform`` returns ``[0, 1)``, so ``u`` can be exactly ``0`` where
    ``math.log`` raises. ``1 - u`` is uniform too and ``log1p`` is defined on
    precisely the range ``uniform()`` guarantees. On the torch backend this is
    not academic: ``torch.rand`` is float32, so ``P(u == 0)`` is ``2^-24``.

See :mod:`specdiff.verifiers.dgrs` for Algorithm 2, which relaxes ``K = 1``.
"""

from __future__ import annotations

import math

from ..types import VerifyRequest, VerifyResult
from ..verify import Verifier, register_verifier
from .rank1 import Rank1Frame


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
