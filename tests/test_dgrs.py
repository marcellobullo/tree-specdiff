"""Tests for Algorithm 2, diffusion greedy rejection sampling. `python tests/test_dgrs.py`.

Same four obligations as `test_rmc.py`, plus the two that only exist once `K > 1`:
the sweep must respect drafting order, and the default residual branch must
carry `Z_perp,1`. Configurable residual complements must preserve the joint
Gaussian target law.

The acceptance probability is checked against Theorem 2 rather than a constant,
and the residual sampler is tested on its own, since inverting eq. (13) by
bisection is the one piece of numerics in the rule that a KS test on the final
state would only catch indirectly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

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
from specdiff.verifiers.dgrs import GreedyRejectionSampling, _sample_residual  # noqa: E402
from specdiff.verifiers.rank1 import Rank1Frame  # noqa: E402

DIM = 8
SIGMA = 0.7


def _levels(delta, num_children):
    """The recursion of lines 13-15: returns `(lambda_K, G_{K+1})`."""
    level, mass = 0.0, 1.0
    for _ in range(num_children):
        level += mass
        tau = math.log(level) / delta
        mass = standard_normal_sf(tau - delta / 2.0) - level * standard_normal_sf(
            tau + delta / 2.0
        )
    return level, mass


def _predicted_acceptance(delta, num_children):
    """Theorem 2, eqs. (14)-(15): `P(exists k : Y = Y_k) = 1 - G_{K+1}`."""
    return 1.0 - _levels(delta, num_children)[1]


def _request(delta, rng, num_children=4, dim=DIM, sigma=SIGMA, scalars=None):
    """One node with `num_children` drafted children, in drafting order.

    `scalars` pins the children's projected coordinates instead of drawing them,
    which is what makes the ordering test deterministic.
    """
    direction = np.zeros(dim)
    direction[0] = 1.0
    mu_p = np.zeros(dim)
    mu_q = mu_p + sigma * delta * direction
    if scalars is None:
        children = mu_p + sigma * rng.standard_normal((num_children, dim))
    else:
        children = np.stack([mu_p + sigma * s * direction for s in scalars])
    return VerifyRequest(
        step=0,
        proposal_mean=mu_p,
        target_mean=mu_q,
        sigma=sigma,
        children=children,
        parent_state=mu_p,
        rng=rng,
    )


def _ks_uniform(samples):
    """Two-sided one-sample KS statistic against Uniform[0, 1]."""
    xs = sorted(float(x) for x in samples)
    n = len(xs)
    return max(max(x - i / n, (i + 1) / n - x) for i, x in enumerate(xs))


# ------------------------------------------------------------------- exactness
def test_exactness_across_delta_and_k():
    """The obligation the sampler cannot check per call, over the grid that
    actually varies: `delta = 0` is the case Remark 2 forces every rule to
    special-case, and `K` is the axis Algorithm 2 exists to exploit."""
    for delta in (0.0, 1e-9, 0.1, 1.0, 3.0):
        for num_children in (1, 2, 4, 8):
            report = check_exactness(
                GreedyRejectionSampling(),
                delta=delta,
                num_children=num_children,
                seed=0,
                alpha=0.001,
            )
            assert report.passed, report


def test_exactness_in_float32():
    report = check_exactness(
        GreedyRejectionSampling(),
        delta=1.0,
        num_children=4,
        seed=0,
        alpha=0.001,
        array_like=np.zeros(DIM, dtype=np.float32),
    )
    assert report.passed, report


def test_degenerate_delta_always_accepts_the_first_child():
    """Remark 2. `beta_1 = 1 ^ rho(S_1) / G_1` is 1 when the kernels coincide,
    so the sweep never gets past the first child."""
    rng = np.random.default_rng(0)
    for delta in (0.0, 1e-16, 1e-12):
        for _ in range(50):
            result = GreedyRejectionSampling().verify(_request(delta, rng))
            assert result.accepted
            assert result.child_index == 0
            assert result.proposals_examined == 1


# ------------------------------------------------------------------ acceptance
def test_acceptance_rate_matches_theorem_2():
    """Verify ``1 - G_{K+1}`` from Equations 14--15 across values of ``K``."""
    n = 20000
    for delta in (0.25, 1.0, 3.0):
        for num_children in (1, 2, 4, 8):
            report = check_exactness(
                GreedyRejectionSampling(),
                delta=delta,
                num_children=num_children,
                seed=1,
                num_samples=n,
            )
            predicted = _predicted_acceptance(delta, num_children)
            stderr = math.sqrt(predicted * (1.0 - predicted) / n)
            assert abs(report.acceptance_rate - predicted) < 4.0 * stderr, (
                f"delta={delta} K={num_children}: measured {report.acceptance_rate:.4f}, "
                f"predicted {predicted:.4f}, {stderr:.4f} se"
            )


def test_single_child_collapses_to_equation_16():
    """At `K = 1` the recursion gives `lambda_1 = 1`, so acceptance is
    `2 * Phi_bar(delta / 2)` -- the same as RMC (the note under eq. 16). The two
    rules differ at `K = 1` only in what they return on rejection."""
    for delta in (0.25, 1.0, 3.0):
        assert math.isclose(
            _predicted_acceptance(delta, 1), 2.0 * standard_normal_sf(delta / 2.0), rel_tol=1e-12
        )


def test_acceptance_increases_with_width():
    """Theorem 2's consequence and the reason the rule exists: extra proposals
    per node buy acceptance, which is what RMC's chain cannot do."""
    for delta in (0.25, 1.0, 3.0):
        rates = [_predicted_acceptance(delta, k) for k in (1, 2, 4, 8)]
        assert all(a < b for a, b in zip(rates, rates[1:])), rates


# ------------------------------------------------------------ sequence coupling
def test_children_are_examined_in_drafting_order():
    """The sequence coupling, not the list coupling.

    Child 0 is placed above `delta / 2`, where `rho >= 1 = G_1` forces
    `beta_1 = 1`, and a far better child is placed at index 2. A rule that
    sorted the children, or scanned them by likelihood ratio, would take index 2.
    """
    rng = np.random.default_rng(0)
    delta = 1.0
    request = _request(delta, rng, num_children=3, scalars=[delta / 2.0 + 0.5, -3.0, 12.0])
    for _ in range(200):
        result = GreedyRejectionSampling().verify(request)
        assert result.accepted
        assert result.child_index == 0, "a later child was preferred: ordering was not respected"
        assert result.proposals_examined == 1


def test_accepted_state_is_the_drafted_child():
    """Obligation 2, plus the telemetry: `proposals_examined` is the 1-based
    round that accepted, i.e. `child_index + 1`."""
    rng = np.random.default_rng(1)
    rule, accepts = GreedyRejectionSampling(), 0
    for _ in range(2000):
        request = _request(1.0, rng, num_children=4)
        result = rule.verify(request)
        if not result.accepted:
            assert result.child_index is None
            assert result.proposals_examined == 5
            continue
        accepts += 1
        assert np.array_equal(result.state, request.child(result.child_index))
        assert result.proposals_examined == result.child_index + 1
    assert accepts > 100, f"only {accepts} accepts; the branch is barely covered"


def test_residual_branch_carries_the_first_childs_orthogonal_residual():
    """Line 18 returns `Z_perp,1`, not `Z_perp,K`.

    The sweep's decisions depend only on the projected scalars, so `Z_perp,1` is
    independent of reaching this branch; the last child's is not the one the
    algorithm specifies. Getting this wrong still yields a plausible Gaussian.
    """
    rng = np.random.default_rng(0)
    rule, rejections = GreedyRejectionSampling(), 0
    for _ in range(2000):
        request = _request(3.0, rng, num_children=4)
        frame = Rank1Frame.from_request(request)
        result = rule.verify(request)
        if result.accepted:
            continue
        rejections += 1
        _, z_out = frame.project(result.state)
        _, z_first = frame.project(request.child(0))
        assert np.max(np.abs(z_out - z_first)) < 1e-12
        for j in range(request.num_children):
            assert not np.allclose(result.state, request.child(j)), "residual returned a child"
    assert rejections > 100, f"only {rejections} rejections; the branch is barely covered"


# --------------------------------------------------------- the residual sampler
def test_residual_sampler_inverts_equation_13():
    """`_sample_residual` on its own, against the CDF it claims to invert.

    If `S ~ r_{K+1}` then `F(S)` is uniform, so this is a KS test on the
    bisection itself rather than on the rule's output -- a bias here would
    otherwise only show up diluted by the accepted branch.
    """
    delta, num_children, n = 2.0, 4, 4000
    frame = Rank1Frame.from_request(_request(delta, np.random.default_rng(0)))
    level, mass = _levels(delta, num_children)
    threshold = math.log(level) / delta + delta / 2.0

    def cdf(s):
        q_part = standard_normal_sf(threshold - delta) - standard_normal_sf(s - delta)
        p_part = standard_normal_sf(threshold) - standard_normal_sf(s)
        return (q_part - level * p_part) / mass

    rng = np.random.default_rng(0)
    values = []
    for _ in range(n):
        s = _sample_residual(frame, level, mass, float(rng.random()))
        assert s >= threshold - 1e-9, "residual sampled below its support"
        values.append(cdf(s))

    critical = math.sqrt(-0.5 * math.log(0.001 / 2.0)) / math.sqrt(n)
    statistic = _ks_uniform(values)
    assert statistic <= critical, f"KS={statistic:.4f} > {critical:.4f}"


# ---------------------------------------------------------------- the topology
def test_accepts_any_branching_factor():
    """`max_children = None`: unlike RMC, Algorithm 2 is the rule a tree is for."""
    rule = GreedyRejectionSampling()
    assert rule.max_children is None
    for tree in (DraftTree.chain(3), DraftTree.uniform(4, 3), DraftTree.from_widths([2, 3])):
        rule.check_topology(tree)


def test_registered_under_d_grs_on_a_bare_import():
    assert isinstance(create_verifier("d-grs"), GreedyRejectionSampling)


# ------------------------------------------------------------------ end to end
class _ShiftKernel(TargetTransition):
    def means(self, indices_in_batch, states, steps):
        return states + 0.1

    # A translation-like toy map: the increment is the thing to freeze.
    def freeze_drift(self, states, means, steps):
        return means - states

    def apply_drift(self, drift, states, steps):
        return states + drift


def test_runs_end_to_end_on_a_branching_tree():
    target = _ShiftKernel()
    sampler = SpeculativeSampler(
        target=target,
        proposal=DelayedDriftProposal(target),
        schedule=ConstantSchedule(0.5),
        tree=DraftTree.uniform(branching=3, lookahead=2),
        verifier=GreedyRejectionSampling(),
        num_steps=20,
        check_contract=True,
    )
    result = sampler.sample(np.zeros(DIM), rng=np.random.default_rng(0))

    assert result.trajectory.shape == (21, DIM)
    assert np.isfinite(result.trajectory).all()
    assert result.target_calls <= 20
    assert 0.0 <= result.acceptance_rate <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


@pytest.mark.parametrize("mode", ["first", "fresh", "nearest_projection"])
def test_configurable_residual_joint_law(mode):
    from scipy.stats import kstest
    from experiments.verifier_config import configured_verifier

    rule = configured_verifier("d-grs", {"d-grs": {"residual_complement": mode}})
    rng = np.random.default_rng(91)
    outputs, residuals = [], 0
    for _ in range(1800):
        request = _request(2., rng, dim=3, sigma=1.)
        frame = Rank1Frame.from_request(request)
        result = rule(request)
        if not result.accepted:
            residuals += 1
            s, perp = frame.project(result.state)
            projections = [frame.project(request.child(j)) for j in range(request.num_children)]
            if mode == "nearest_projection":
                j = min(range(request.num_children), key=lambda j: abs(projections[j][0] - s))
                assert np.allclose(perp, projections[j][1], atol=1e-12, rtol=0)
            elif mode == "fresh":
                assert all(not np.array_equal(perp, p) for _, p in projections)
        outputs.append(result.state - request.target_mean)
    x = np.array(outputs)
    assert residuals > 100
    assert all(kstest(x[:, j], "norm").statistic < .055 for j in range(3))
    assert np.max(np.abs(np.cov(x.T) - np.eye(3))) < .13
    assert kstest(x.sum(axis=1) / np.sqrt(3), "norm").statistic < .055


def test_invalid_residual_complement():
    with pytest.raises(ValueError, match="residual_complement"):
        GreedyRejectionSampling(residual_complement="nearest_vector")
