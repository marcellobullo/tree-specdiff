"""Verification rules and the coordinates they share.

:mod:`~specdiff.verifiers.rank1` holds the rank-1 reduction of eqs. (8)-(11),
which belongs to the template rather than to any one rule.
:mod:`~specdiff.verifiers.stubs` is where Algorithms 1 and 2 go.
"""

from __future__ import annotations

from .rank1 import DEGENERATE_TOL, Rank1Frame
from .stubs import GreedyRejectionSampling, ReflectionMaximalCoupling

__all__ = [
    "Rank1Frame",
    "DEGENERATE_TOL",
    "ReflectionMaximalCoupling",
    "GreedyRejectionSampling",
]
