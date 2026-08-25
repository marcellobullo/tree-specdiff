"""Tests for the rank-1 reduction. `python tests/test_rank1.py`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdiff import VerifyRequest  # noqa: E402
from specdiff.ops import standard_normal_cdf, standard_normal_sf  # noqa: E402
from specdiff.verifiers.rank1 import (  # noqa: E402
    DEFAULT_DEGENERATE_TOL,
    Rank1Frame,
)


def _frame(delta, dim=8, dtype=np.float64, sigma=1.0, tol=None):
    direction = np.zeros(dim, dtype=dtype)
    direction[0] = 1.0
    mu_p = np.zeros(dim, dtype=dtype)
    mu_q = (mu_p + sigma * delta * direction).astype(dtype)
    request = VerifyRequest(
        step=0,
        proposal_mean=mu_p,
        target_mean=mu_q,
        sigma=sigma,
        children=np.zeros((2, dim), dtype=dtype),
    )
    return Rank1Frame.from_request(request, tol=tol)


def test_delta_and_direction():
    f = _frame(1.5)
    assert math.isclose(f.delta, 1.5, rel_tol=1e-12)
    assert math.isclose(float(np.linalg.norm(f.direction)), 1.0, rel_tol=1e-12)
    assert not f.degenerate


def test_project_reconstruct_round_trips():
    f = _frame(0.8)
    y = np.random.default_rng(0).standard_normal(8)
    s, z_perp = f.project(y)
    assert np.allclose(f.reconstruct(s, z_perp), y)
    # the orthogonal part really is orthogonal
    assert abs(float(np.dot(z_perp, f.direction))) < 1e-12


def test_projection_law_is_the_scalar_reduction():
    """Y ~ N(mu_q, sigma^2 I)  =>  S ~ N(delta, 1). This is the whole point."""
    delta, sigma, n = 1.3, 0.7, 20000
    f = _frame(delta, sigma=sigma)
    rng = np.random.default_rng(1)
    mu_q = f.mu_p + sigma * delta * f.direction
    ys = mu_q + sigma * rng.standard_normal((n, 8))
    ss = np.array([f.project(y)[0] for y in ys])
    assert abs(ss.mean() - delta) < 4 / math.sqrt(n)
    assert abs(ss.std() - 1.0) < 0.05


# ------------------------------------------------------------------ degeneracy
def test_degenerate_uses_a_tolerance_not_exact_equality():
    """The regime that breaks a rule is small-and-nonzero delta, which is
    exactly what a *good* proposal produces. An `== 0.0` test never fires
    there and leaves tau = ln(lambda)/delta to blow up."""
    assert _frame(0.0).degenerate
    assert _frame(1e-16).degenerate
    assert _frame(1e-12).degenerate
    assert not _frame(1e-3).degenerate


def test_degenerate_tolerance_does_not_follow_the_dtype():
    """The threshold is a constant, deliberately.

    Everything that can break down here -- `delta`, `tau`, the D-GRS masses --
    is a Python float computed in float64 whatever the states carry, so the
    floor is a property of float64, not of the array. It used to be `sqrt(eps)`
    of the state dtype, which is ~3.1e-2 in float16: large enough that
    accepting unconditionally admits 1.2% total variation per node, in exactly
    the small-delta regime a good proposal produces.
    """
    for dtype in (np.float64, np.float32, np.float16):
        assert _frame(1.0, dtype=dtype).tol == DEFAULT_DEGENERATE_TOL
    # A displacement float16 *can* represent is not degenerate any more. Under
    # the old sqrt(eps) rule this was, and the coupling silently skipped its
    # acceptance test here.
    assert not _frame(1e-5, dtype=np.float16).degenerate
    # Below the dtype's own smallest subnormal (~6e-8) the two means are
    # bit-identical, so delta really is zero. That is the dtype's limit, not
    # the tolerance's, and firing the shortcut there is correct.
    assert _frame(1e-9, dtype=np.float16).degenerate


def test_degenerate_tolerance_clears_the_float64_floor():
    """Pins the mechanism the constant is sized against, so it cannot be
    retuned blind.

    `G_2 = Phi_bar(-delta/2) - Phi_bar(delta/2)` is mathematically
    `delta / sqrt(2 pi)`. float64 resolves it until the two survival terms
    cancel outright, at `delta ~ 1e-15`. Below that the D-GRS residual mass is
    zero and the rule takes its fallback path, so the tolerance must sit above
    it -- with margin, since the only cost of a *smaller* tolerance is that the
    shortcut fires less often.
    """
    floor = None
    for exponent in range(10, 20):
        delta = 10.0**-exponent
        if standard_normal_sf(-delta / 2) - standard_normal_sf(delta / 2) <= 0.0:
            floor = delta
            break
    assert floor is not None, "G_2 never underflowed; the floor moved"
    assert floor <= 1e-15, f"floor at {floor:.0e}, tighter than assumed"
    assert DEFAULT_DEGENERATE_TOL > floor * 1e4, "tolerance has too little margin"
    # ... and the shortcut it buys costs almost nothing.
    assert 2.0 * standard_normal_cdf(DEFAULT_DEGENERATE_TOL / 2) - 1.0 < 1e-10


def test_tau_is_guarded_on_a_degenerate_frame():
    f = _frame(1e-16)
    try:
        f.tau(0.5)
    except ZeroDivisionError as exc:
        assert "degenerate" in str(exc)
    else:
        raise AssertionError("tau returned a value on a degenerate frame")


def test_tau_matches_the_closed_form_when_well_conditioned():
    f = _frame(1.25)
    assert math.isclose(f.tau(0.5), math.log(0.5) / 1.25, rel_tol=1e-12)
    try:
        f.tau(0.0)
    except ValueError as exc:
        assert "ln(level)" in str(exc)
    else:
        raise AssertionError("tau(0) should be rejected")


def test_explicit_tolerance_overrides_the_dtype_default():
    assert _frame(1e-3, tol=1e-2).degenerate
    assert not _frame(1e-3, tol=1e-6).degenerate


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
