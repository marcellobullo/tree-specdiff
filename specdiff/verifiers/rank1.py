"""The rank-1 reduction of eqs. (8)-(11).

This module implements the coordinate transformation from Section 3, which is
shared by RMC and D-GRS. Because
``P = N(mu_p, sigma^2 I)`` and ``Q = N(mu_q, sigma^2 I)`` differ only in their
means, everything orthogonal to ``mu_q - mu_p`` has the same law under both,
and the d-dimensional coupling collapses to a scalar one::

    S ~ N(0, 1)      under P
    S ~ N(delta, 1)  under Q

A rule therefore only has to move the scalar ``S``; the orthogonal residual of
whichever proposal it keeps is carried through untouched.

Centralizing the transformation keeps projection and reconstruction consistent
across verification rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..ops import Backend, resolve_backend
from ..types import VerifyRequest

Array = Any

DEGENERATE_TOL: Optional[float] = None
"""Default tolerance below which ``delta`` counts as zero.

``None`` means :data:`DEFAULT_DEGENERATE_TOL`. Set a float here to override
globally, or pass ``tol=`` to :meth:`Rank1Frame.from_request`.
"""

DEFAULT_DEGENERATE_TOL: float = 1e-10
"""Constant degeneracy threshold, independent of the state dtype.

Every quantity that can break down here -- ``delta``, ``tau = ln(lambda)/delta``
and the D-GRS masses ``G_k`` -- is a Python float, computed in float64 no matter
what dtype the states carry (``ops.norm`` and ``ops.dot`` return floats). So the
floor is a property of float64 arithmetic, not of the state array.

That floor is ``delta ~ 1e-15``: ``G_2 = Phi_bar(-delta/2) - Phi_bar(delta/2)``
is mathematically ``delta / sqrt(2 pi)``, and float64 resolves it down to about
``4e-16`` before the two survival terms cancel to exactly zero. ``1e-10`` clears
that by five orders of magnitude while admitting only ``~4e-11`` of total
variation (the shortcut's cost is ``TV = delta / sqrt(2 pi)``).

The frame is self-consistent by construction: ``delta * direction == diff``.
A constant threshold therefore avoids dtype-dependent bias in the
small-``delta`` regime.
"""


@dataclass(frozen=True)
class Rank1Frame:
    """Coordinates induced by the normalised mean displacement (eq. 8).

    Attributes
    ----------
    delta:
        ``||mu_q - mu_p|| / sigma``, the normalised mean mismatch. This single
        number controls every acceptance probability in the paper: eq. (16) for
        RMC and eq. (15) for D-GRS.
    direction:
        The unit vector ``e``.
    """

    delta: float
    direction: Array
    mu_p: Array
    sigma: float
    ops: Backend
    tol: float = 0.0
    """Tolerance defining :attr:`degenerate`; see :data:`DEGENERATE_TOL`."""

    @classmethod
    def from_request(cls, request: VerifyRequest, *, tol: Optional[float] = None) -> "Rank1Frame":
        ops = request.backend or resolve_backend(request.children)
        diff = (request.target_mean - request.proposal_mean) / request.sigma
        delta = ops.norm(diff)
        direction = diff / delta if delta > 0.0 else diff
        if tol is None:
            tol = DEGENERATE_TOL
        if tol is None:
            tol = DEFAULT_DEGENERATE_TOL
        return cls(
            delta=delta,
            direction=direction,
            mu_p=request.proposal_mean,
            sigma=request.sigma,
            ops=ops,
            tol=float(tol),
        )

    @property
    def degenerate(self) -> bool:
        """Return whether ``delta`` is zero to working precision.

        Remark 2: the acceptance probability is 1 for every ``K``, and
        ``tau = ln(lambda) / delta`` is undefined, so rules must special-case
        this rather than divide by zero.

        The test is ``delta <= tol``, **not** ``delta == 0``: ``tol`` is a
        small constant (:data:`DEFAULT_DEGENERATE_TOL`) sitting above the
        float64 floor where ``G_2`` cancels to zero, so the branch fires only
        where the two kernels really are the same to working precision.

        Note that neither of the paper's rules actually *needs* this branch
        above that floor -- both were measured exact with it disabled at every
        ``delta`` down to and including exactly ``0``. ``lambda_1 = 1``, so
        ``tau_1 = ln(1)/delta = 0`` for any ``delta > 0``, and ``lambda`` then
        grows only by ``G ~ delta``, which keeps ``ln(lambda)/delta`` at O(1)
        rather than blowing up. The branch is a guard on the residual path at
        ``delta ~ 1e-16``, and a short-circuit; it is not what makes small
        ``delta`` safe.

        The shortcut accepts unconditionally and introduces
        ``TV = delta / sqrt(2 pi)`` whenever used. Keep the tolerance close to
        the numerical floor to limit this approximation.
        """
        return self.delta <= self.tol

    def tau(self, level: float) -> float:
        """``tau = ln(level) / delta`` (Appendix B.2), guarded.

        The threshold whose half-space masses give the super-level set masses
        ``Q(H) = Phi_bar(tau - delta/2)`` and ``P(H) = Phi_bar(tau + delta/2)``.
        Raises on a degenerate frame to prevent an infinite threshold from
        propagating into a zero residual mass.
        """
        if self.degenerate:
            raise ZeroDivisionError(
                f"tau is undefined on a degenerate frame (delta={self.delta:.3e} "
                f"<= tol={self.tol:.3e}). Check `frame.degenerate` first and accept "
                "the first proposal: Remark 2 gives acceptance probability 1."
            )
        if level <= 0.0:
            raise ValueError(f"level must be > 0 for ln(level); got {level}")
        return math.log(level) / self.delta

    def project(self, state: Array) -> Tuple[float, Array]:
        """``Y -> (S, Z_perp)`` of eq. (10)."""
        z = (state - self.mu_p) / self.sigma
        s = self.ops.dot(self.direction, z)
        z_perp = z - s * self.direction
        return s, z_perp

    def reconstruct(self, s: float, z_perp: Array) -> Array:
        """``(S, Z_perp) -> Y`` of eq. (11)."""
        return self.mu_p + self.sigma * (s * self.direction + z_perp)


RESIDUAL_COMPLEMENTS = ("first", "fresh", "nearest_projection")


def residual_complement(frame, request, s, policy, *, projections=None):
    """Choose the orthogonal noise independently of its values.

    Selection by scalar projection preserves the Gaussian perpendicular law;
    selecting by full-vector distance generally does not.
    """
    if policy not in RESIDUAL_COMPLEMENTS:
        raise ValueError(f"residual_complement must be one of {RESIDUAL_COMPLEMENTS}")
    if policy == "fresh":
        noise = frame.ops.randn_stack(1, request.proposal_mean, request.rng)[0]
        return noise - frame.ops.dot(frame.direction, noise) * frame.direction
    if projections is None:
        projections = [frame.project(request.child(j)) for j in range(request.num_children)]
    j = (min(range(len(projections)), key=lambda j: abs(projections[j][0] - s))
         if policy == "nearest_projection" else 0)
    return projections[j][1]
