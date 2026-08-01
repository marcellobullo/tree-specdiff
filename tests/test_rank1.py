"""Tests for the rank-1 reduction. `python tests/test_rank1.py`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdiff import VerifyRequest  # noqa: E402
from specdiff.verifiers.rank1 import Rank1Frame  # noqa: E402


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


def test_degenerate_tolerance_follows_the_dtype():
    """sqrt(eps) is ~1.5e-8 in float64 but ~3.4e-4 in float32; a single
    hardcoded constant would be wrong for one of them."""
    f64, f32 = _frame(1e-5, dtype=np.float64), _frame(1e-5, dtype=np.float32)
    assert f64.tol < 1e-6 < f32.tol
    assert not f64.degenerate
    assert f32.degenerate


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
