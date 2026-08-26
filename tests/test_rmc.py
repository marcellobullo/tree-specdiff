"""Tests for Algorithm 1, the reflection maximal coupling. `python tests/test_rmc.py`.

Four things are worth asserting about a coupling, and only the first is about
exactness:

1.  the returned state is an exact draw from ``N(mu_q, sigma^2 I)`` -- what
    :func:`specdiff.check_exactness` tests, and the reason the method is sound;
2.  it accepts as often as eq. (16) says -- what makes it *maximal* rather than
    merely exact. ``ResampleVerifier`` passes (1) and fails (2), so without this
    the suite would not distinguish RMC from a rule that never accepts;
3.  the rejection branch really is the reflection, carrying the child's
    orthogonal residual through untouched -- structural, and the failure mode
    that survives a casual eyeball check;
4.  it refuses a branching tree at construction time.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdiff import (  # noqa: E402
    DelayedDriftProposal,
    DraftTree,
    SpeculativeSampler,
    VerifyRequest,
    check_exactness,
    create_verifier,
)
from specdiff.kernels import ConstantSchedule, TargetTransition  # noqa: E402
from specdiff.ops import standard_normal_sf  # noqa: E402
from specdiff.verifiers.rank1 import Rank1Frame  # noqa: E402
from specdiff.verifiers.rmc import ReflectionMaximalCoupling  # noqa: E402

DIM = 8
SIGMA = 0.7


def _request(delta, rng, dim=DIM, sigma=SIGMA, dtype=np.float64):
    """One K = 1 node whose kernels differ by `delta` along the first axis."""
    direction = np.zeros(dim, dtype=dtype)
    direction[0] = 1.0
    mu_p = np.zeros(dim, dtype=dtype)
    mu_q = (mu_p + sigma * delta * direction).astype(dtype)
    child = (mu_p + sigma * rng.standard_normal(dim)).astype(dtype)
    return VerifyRequest(
        step=0,
        proposal_mean=mu_p,
        target_mean=mu_q,
        sigma=sigma,
        children=child[None, :],
        parent_state=mu_p,
        rng=rng,
    )


# ------------------------------------------------------------------- exactness
def test_exactness_across_delta():
    """Verify distributional exactness across representative `delta` values.

    `alpha=0.001` rather than the default: a sweep runs many tests, so at
    `alpha=0.01` the chance that some cell trips on a correct rule grows with
    the number of cells. The seed is fixed for the same reason -- this is a
    hypothesis test, so a correct rule fails it about `alpha` of the time, and
    an irreproducible failure is indistinguishable from a coupling bug.
    """
    for delta in (0.0, 1e-9, 0.1, 1.0, 3.0):
        report = check_exactness(
            ReflectionMaximalCoupling(), delta=delta, num_children=1, seed=0, alpha=0.001
        )
        assert report.passed, report


def test_exactness_in_float32():
    """The rank-1 reduction takes a norm and divides by it; `degenerate` uses
    `sqrt(eps)` of the state dtype, which differs by four orders of magnitude
    between float32 and float64."""
    report = check_exactness(
        ReflectionMaximalCoupling(),
        delta=1.0,
        num_children=1,
        seed=0,
        alpha=0.001,
        array_like=np.zeros(DIM, dtype=np.float32),
    )
    assert report.passed, report


def test_degenerate_delta_always_accepts():
    """Remark 2: below tolerance the kernels are indistinguishable at the
    state's own precision, so acceptance probability is 1. Accepting is the
    correct limit, not an approximation."""
    rng = np.random.default_rng(0)
    for delta in (0.0, 1e-16, 1e-12):
        for _ in range(50):
            result = ReflectionMaximalCoupling().verify(_request(delta, rng))
            assert result.accepted
            assert result.child_index == 0


# ------------------------------------------------------------------ maximality
def test_acceptance_rate_matches_equation_16():
    """`2 * Phi_bar(delta / 2)`, the overlap `1 - TV(P, Q)`.

    This is what separates RMC from *an* exact rule: `ResampleVerifier` passes
    every exactness test above and accepts nothing. Note the sign -- the
    survival function is evaluated at `+delta/2`; `2 * Phi_bar(-delta / 2)`
    exceeds 1 for every delta > 0.
    """
    n = 20000
    for delta in (0.1, 0.5, 1.0, 2.0, 3.0):
        report = check_exactness(
            ReflectionMaximalCoupling(), delta=delta, num_children=1, seed=1, num_samples=n
        )
        predicted = 2.0 * standard_normal_sf(delta / 2.0)
        stderr = math.sqrt(predicted * (1.0 - predicted) / n)
        assert abs(report.acceptance_rate - predicted) < 4.0 * stderr, (
            f"delta={delta}: measured {report.acceptance_rate:.4f}, "
            f"predicted {predicted:.4f}, {stderr:.4f} se"
        )


def test_proposals_examined_is_always_one():
    """The third return value of Algorithm 1. A single-proposal coupling looks
    at its one child whichever branch it takes."""
    rng = np.random.default_rng(3)
    for delta in (0.0, 1.0, 3.0):
        for _ in range(50):
            assert ReflectionMaximalCoupling().verify(_request(delta, rng)).proposals_examined == 1


# ------------------------------------------------------------- the reflection
def test_rejection_reflects_and_keeps_the_orthogonal_residual():
    """Lines 9 and 18 of the algorithm, in rank-1 coordinates.

    On rejection the scalar becomes `delta - s_hat` and `Z_perp` is the
    *child's*, unmodified. Carrying the wrong residual produces samples that
    look Gaussian and fail `check_exactness` -- this pins down which one.
    """
    rng = np.random.default_rng(0)
    rule, rejections = ReflectionMaximalCoupling(), 0
    for _ in range(2000):
        request = _request(2.5, rng)
        frame = Rank1Frame.from_request(request)
        s_hat, z_perp = frame.project(request.child(0))

        result = rule.verify(request)
        if result.accepted:
            continue
        rejections += 1
        s_out, z_out = frame.project(result.state)
        assert abs(s_out - (frame.delta - s_hat)) < 1e-12
        assert np.max(np.abs(z_out - z_perp)) < 1e-12

    assert rejections > 100, f"only {rejections} rejections; the branch is barely covered"


def test_accepted_state_is_the_drafted_child():
    """Obligation 2. The sampler descends into the accepted child's subtree, so
    a state that merely resembles it sends the trajectory down the wrong branch.
    Note `array_equal`, not `allclose`: a reconstructed state would pass
    `CheckedVerifier` and still be the wrong object."""
    rng = np.random.default_rng(1)
    rule, accepts = ReflectionMaximalCoupling(), 0
    for _ in range(2000):
        request = _request(0.5, rng)
        result = rule.verify(request)
        if not result.accepted:
            assert result.child_index is None
            continue
        accepts += 1
        assert result.child_index == 0
        assert np.array_equal(result.state, request.child(0))

    assert accepts > 100, f"only {accepts} accepts; the branch is barely covered"


def test_both_endpoints_of_the_uniform_are_handled():
    """`ops.uniform` returns [0, 1), so `u` can be exactly 0 -- where `math.log`
    raises `ValueError` and `math.log1p(-u)` does not. One node in 2^53 on NumPy, and
    one in 2^24 (~6e-8) on the torch backend, where torch.rand is float32,
    i.e. never in this suite and eventually in a long run.

    Note which endpoint is which. Comparing `log1p(-u)` means the effective
    uniform is `1 - u`, so `u = 0` is the *least* likely draw to accept, not the
    most: the flip that makes `math.log(u) -> -inf` accept unconditionally goes
    the other way here. Both forms are correct in law; only the mapping from a
    given `u` to a decision differs. So this asserts the decision the rule
    should make at each endpoint rather than assuming either one accepts.
    """

    class FixedRng:
        def __init__(self, u):
            self._u = u

        def random(self):
            return self._u

    base = _request(2.0, np.random.default_rng(0))
    frame = Rank1Frame.from_request(base)
    s_hat, _ = frame.project(base.child(0))
    log_ratio = frame.delta * (s_hat - 0.5 * frame.delta)

    for u in (0.0, float(np.nextafter(1.0, 0.0))):
        request = VerifyRequest(
            step=0,
            proposal_mean=base.proposal_mean,
            target_mean=base.target_mean,
            sigma=base.sigma,
            children=base.children,
            rng=FixedRng(u),
        )
        result = ReflectionMaximalCoupling().verify(request)  # must not raise
        assert result.accepted == (math.log1p(-u) <= log_ratio), f"u={u!r}"


# ---------------------------------------------------------------- the topology
def test_refuses_a_branching_tree():
    """Reject branching trees for a verifier with `max_children = 1`."""
    ReflectionMaximalCoupling().check_topology(DraftTree.chain(4))
    try:
        ReflectionMaximalCoupling().check_topology(DraftTree.uniform(branching=3, lookahead=2))
    except ValueError as exc:
        assert "at most K=1" in str(exc)
    else:
        raise AssertionError("a branching tree should have been refused")


def test_registered_under_rmc_on_a_bare_import():
    """`create_verifier` is what a config file addresses, and registration is a
    side effect of importing `specdiff.verifiers` -- which `specdiff/__init__`
    must therefore do eagerly."""
    assert isinstance(create_verifier("rmc"), ReflectionMaximalCoupling)


# --------------------------------------------------------------- end to end
class _ShiftKernel(TargetTransition):
    """A trivial target: one fixed step per call. Enough to drive the sampler."""

    def means(self, indices_in_batch, states, steps):
        return states + 0.1


def test_runs_end_to_end_under_the_contract_checker():
    """A whole trajectory with `check_contract=True`, which enforces shape,
    finiteness, index bounds and accepted-state identity at every node."""
    target = _ShiftKernel()
    sampler = SpeculativeSampler(
        target=target,
        proposal=DelayedDriftProposal(target),
        schedule=ConstantSchedule(0.5),
        tree=DraftTree.chain(3),
        verifier=ReflectionMaximalCoupling(),
        num_steps=20,
        check_contract=True,
    )
    result = sampler.sample(np.zeros(DIM), rng=np.random.default_rng(0))

    assert result.trajectory.shape == (21, DIM)
    assert np.isfinite(result.trajectory).all()
    assert result.target_calls <= 20, "speculation must not cost more calls than the plain sampler"
    assert 0.0 <= result.acceptance_rate <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
