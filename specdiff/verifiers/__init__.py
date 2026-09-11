"""Verification rules and the coordinates they share.

:mod:`~specdiff.verifiers.rank1` holds the rank-1 reduction of eqs. (8)-(11),
which belongs to the template rather than to any one rule.
:mod:`~specdiff.verifiers.rmc` is Algorithm 1 and :mod:`~specdiff.verifiers.dgrs`
is Algorithm 2. :mod:`~specdiff.verifiers.paws` implements PAWS list coupling.
Importing this package registers ``"rmc"``, ``"d-grs"``, and ``"paws"``.
"""

from __future__ import annotations

from .dgrs import GreedyRejectionSampling
from .paws import RankSelectionCoupling
from .rank1 import DEGENERATE_TOL, Rank1Frame
from .rmc import ReflectionMaximalCoupling

__all__ = [
    "Rank1Frame",
    "DEGENERATE_TOL",
    "ReflectionMaximalCoupling",
    "GreedyRejectionSampling",
    "RankSelectionCoupling",
]
