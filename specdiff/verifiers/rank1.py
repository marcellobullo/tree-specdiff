"""The rank-1 reduction of eqs. (8)-(11).

This is not a coupling. It is the change of coordinates that Section 3
introduces *before* either RMC or D-GRS, and that both then use: because
``P = N(mu_p, sigma^2 I)`` and ``Q = N(mu_q, sigma^2 I)`` differ only in their
means, everything orthogonal to ``mu_q - mu_p`` has the same law under both,
and the d-dimensional coupling collapses to a scalar one::

    S ~ N(0, 1)      under P
    S ~ N(delta, 1)  under Q

A rule therefore only has to move the scalar ``S``; the orthogonal residual of
whichever proposal it keeps is carried through untouched.

Provided here because it belongs to the template, not to any one rule, and
because getting the reconstruction wrong is a silent-correctness bug: the
sample still looks Gaussian, just not from the right distribution.
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

``None`` means "derive it from the state dtype" -- ``sqrt(eps)``, which is
~1.5e-8 in float64 and ~3.4e-4 in float32. Set a float here to override
globally, or pass ``tol=`` to :meth:`Rank1Frame.from_request`.
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
        ops = resolve_backend(request.children)
        diff = (request.target_mean - request.proposal_mean) / request.sigma
        delta = ops.norm(diff)
        direction = diff / delta if delta > 0.0 else diff
        if tol is None:
            tol = DEGENERATE_TOL
        if tol is None:
            tol = math.sqrt(ops.finfo_eps(request.children))
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
        """``delta`` is zero to working precision: accept anything.

        Remark 2: the acceptance probability is 1 for every ``K``, and
        ``tau = ln(lambda) / delta`` is undefined, so rules must special-case
        this rather than divide by zero.

        The test is ``delta <= tol``, **not** ``delta == 0``. Exact equality is
        the wrong predicate here: the regime that breaks a rule is small-and-
        nonzero, and that is exactly what a *good* proposal produces. At
        ``delta = 1e-16`` an exact test says "not degenerate" while
        ``tau = ln(lambda) / delta`` is ~1e15, which saturates ``Phi_bar`` to
        exactly 0 and makes the D-GRS residual mass ``G_k`` vanish -- a
        division by zero one step later. Below ``tol`` the two kernels are
        indistinguishable at the state's own precision anyway (the TV distance
        is ``~delta / sqrt(2 pi)``), so accepting unconditionally is not an
        approximation, it is the correct limit.
        """
        return self.delta <= self.tol

    def tau(self, level: float) -> float:
        """``tau = ln(level) / delta`` (Appendix B.2), guarded.

        The threshold whose half-space masses give the super-level set masses
        ``Q(H) = Phi_bar(tau - delta/2)`` and ``P(H) = Phi_bar(tau + delta/2)``.
        Raises on a degenerate frame rather than returning an infinity that
        would silently propagate into a zero residual mass.
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
        return s, z - s * self.direction

    def reconstruct(self, s: float, z_perp: Array) -> Array:
        """``(S, Z_perp) -> Y`` of eq. (11)."""
        return self.mu_p + self.sigma * (s * self.direction + z_perp)
