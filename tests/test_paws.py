"""Distributional and integration checks for the PAWS verifier."""
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import beta, kstest, norm

from specdiff import (
    BatchedSpeculativeSampler, CheckedVerifier, ConstantSchedule, DraftTree,
    IdentityProposal, RankSelectionCoupling, SpeculativeSampler,
    TargetTransition, VerifyRequest, check_exactness, create_verifier,
)
from specdiff.verifiers.paws import _ResidualCDF, _SelectedDensity, rank_weights


def request(delta=1.0, k=4, seed=42):
    rng = np.random.default_rng(seed)
    return VerifyRequest(
        step=0, proposal_mean=np.zeros(3), target_mean=np.array([delta, 0., 0.]),
        sigma=1., children=rng.standard_normal((k, 3)), rng=rng,
    )


def test_defaults_registration_and_topology_capabilities():
    verifier = create_verifier("paws")
    assert isinstance(verifier, RankSelectionCoupling)
    assert verifier.residual_complement == "first"
    assert verifier.rank_policy == "optimized"
    assert verifier.residual_method == "inverse_cdf"
    tree = DraftTree.uniform(2, 3)
    assert verifier.matched_tree(tree, num_steps=20) is tree
    assert not verifier.requires_chain
    rmc = create_verifier("rmc")
    assert rmc.requires_chain
    assert CheckedVerifier(rmc).requires_chain
    assert rmc.matched_tree(tree, num_steps=20).depth == 7
    assert rmc.matched_tree(tree, num_steps=20, match="budget").depth == 14
    assert rmc.matched_tree(tree, num_steps=20, evaluate_leaves=True).depth == 14


@pytest.mark.parametrize("policy", ["optimized", "uniform", "max"])
@pytest.mark.parametrize("delta", [0., 0.1, 1., 3., 20.])
def test_scalar_target_law(policy, delta):
    result = check_exactness(
        RankSelectionCoupling(rank_policy=policy), delta=delta,
        num_children=4, num_samples=1200, alpha=0.0001, seed=731,
    )
    assert result.passed, result


@pytest.mark.parametrize("policy", ["optimized", "uniform", "max"])
@pytest.mark.parametrize("delta", [0., 0.1, 1., 3., 20.])
def test_scalar_target_law_with_the_rejection_residual(policy, delta):
    """The rejection residual of eq. (14) is the same coupling, sampled anew.

    Exactness is the point: the two residual implementations differ in cost, so
    a rule configured either way has to pass the same distributional test.
    """
    result = check_exactness(
        RankSelectionCoupling(rank_policy=policy, residual_method="rejection"),
        delta=delta, num_children=4, num_samples=1200, alpha=0.0001, seed=731,
    )
    assert result.passed, result


def test_rejection_trial_counters_report_the_cost_and_reset():
    rng = np.random.default_rng(5)
    verifier = RankSelectionCoupling(rank_policy="max", residual_method="rejection")
    base = request(delta=.7, k=4)
    for _ in range(400):
        verifier(replace(base, children=rng.standard_normal((4, 3)), rng=rng))
    assert verifier.residual_draws > 0
    # `1 / total` for this cell; the counter is the realized average.
    expected = 1 / _ResidualCDF(.7, (0., 0., 0., 1.)).total
    assert verifier.residual_trials / verifier.residual_draws == pytest.approx(expected, rel=.25)
    verifier.reset()
    assert (verifier.residual_draws, verifier.residual_trials) == (0, 0)
    # The closed form spends no trials, so the second counter stays at zero.
    closed = RankSelectionCoupling(rank_policy="max")
    for _ in range(50):
        closed(replace(base, children=rng.standard_normal((4, 3)), rng=rng))
    assert closed.residual_draws > 0 and closed.residual_trials == 0


def test_capped_rejection_keeps_the_target_law():
    """A cap of one trial routes almost every correction to the closed form."""
    report = check_exactness(
        RankSelectionCoupling(residual_method="rejection", residual_max_trials=1),
        delta=.3, num_children=4, num_samples=1500, alpha=.0001, seed=204,
    )
    assert report.passed, report


@pytest.mark.parametrize("mode", ["first", "fresh", "nearest_projection"])
def test_joint_law_with_each_complement(mode):
    rng = np.random.default_rng(91)
    verifier = RankSelectionCoupling(residual_complement=mode)
    base = request(delta=2.)
    outputs = []
    for _ in range(1800):
        r = replace(base, children=rng.standard_normal((4, 3)), rng=rng)
        out = verifier(r)
        if out.accepted:
            assert np.array_equal(out.state, r.child(out.child_index))
        outputs.append(out.state)
    x = np.asarray(outputs) - base.target_mean
    assert all(kstest(x[:, j], "norm").statistic < .055 for j in range(3))
    assert np.max(np.abs(np.cov(x.T) - np.eye(3))) < .13
    # Also test a mixed direction, beyond individual marginals.
    assert kstest(x.sum(axis=1) / math.sqrt(3), "norm").statistic < .055


@pytest.mark.parametrize("delta", [.1, 1., 3.])
def test_single_child_acceptance_matches_rmc(delta):
    report = check_exactness(
        RankSelectionCoupling(), delta=delta, num_children=1,
        num_samples=2000, alpha=.0001, seed=84,
    )
    assert report.passed, report
    assert abs(report.acceptance_rate - 2 * ndtr(-delta / 2)) < .04


def test_optimizer_retains_small_gap_and_tail_efficiency():
    for delta, floor in [(0.1, .996), (1., .895), (8., .000124)]:
        w = rank_weights(delta, 4)
        selected = _SelectedDensity(w)
        overlap = quad(
            lambda s: min(norm.pdf(s-delta),
                          norm.pdf(s) * math.exp(selected.log_beta(s))),
            -12, delta + 12, epsabs=1e-10, limit=800,
            points=np.linspace(-12, delta + 12, 101)[1:-1],
        )[0]
        assert overlap > floor
    assert rank_weights(.0001, 4) != rank_weights(0., 4)


@pytest.mark.parametrize("weights", [(0., 0., 0., 1.), (.3, .1, .2, .4)])
def test_inverse_cdf_against_independent_density_integral(weights):
    delta = .7
    residual = _ResidualCDF(delta, weights)
    def density(s):
        g = norm.pdf(s) * sum(
            w * beta.pdf(ndtr(s), r, 5-r)
            for r, w in enumerate(weights, 1)
        )
        return max(0., norm.pdf(s-delta) - g)
    mass = quad(density, -12, 12, points=np.linspace(-10, 10, 81),
                epsabs=1e-11, limit=500)[0]
    assert abs(mass - residual.total) < 1e-8   # `total` is the mass itself now
    for u in [.001, .05, .25, .5, .9, .999]:
        s = residual.sample(u)
        actual = quad(density, -12, s, points=np.linspace(-12, s, 81)[1:-1],
                      epsabs=1e-11, limit=500)[0] / mass
        assert abs(actual - u) < 2e-7
    assert all(math.isfinite(residual.sample(u)) for u in [0., 1.])


@pytest.mark.parametrize("delta", [1e-9, 1e-6])
def test_tiny_gap_uniform_residual_has_the_rayleigh_limit(delta):
    residual = _ResidualCDF(delta, (.25,) * 4)
    for u in [.001, .5, .999]:
        expected = math.sqrt(-2 * math.log1p(-u))
        assert abs(residual.sample(u) - expected) < 2 * delta + 1e-8


def test_extreme_gap_residual_is_target_centered():
    residual = _ResidualCDF(360., (0., 0., 0., 1.))
    for u in [.001, .5, .999]:
        assert abs(residual.sample(u) - (360. + norm.ppf(u))) < 1e-9


def test_disconnected_residual_includes_both_tails():
    residual = _ResidualCDF(.7, (0., 0., 0., 1.))
    assert residual.pdf(-4.) > 0
    assert residual.pdf(0.) == 0
    assert residual.pdf(4.) > 0
    assert residual.sample(.01) < 0
    assert residual.sample(.99) > 2


def test_small_absolute_gap_is_not_mistaken_for_equal_kernels():
    r = request()
    r = replace(r, sigma=2e-6, target_mean=np.array([9e-7, 0, 0]),
                children=np.full((4, 3), -4e-6))
    class RejectRng:
        def random(self):
            return .999
    result = RankSelectionCoupling()(replace(r, rng=RejectRng()))
    assert not result.accepted
    assert result.state[0] > r.target_mean[0]


def test_zero_uniform_and_original_index_after_sorting():
    class ZeroRng:
        def random(self):
            return 0.
    r = replace(request(), children=np.array([[1., 0, 0], [-1., 0, 0], [6., 0, 0], [0., 0, 0]]),
                rng=ZeroRng())
    result = RankSelectionCoupling(rank_policy="max")(r)
    assert result.accepted and result.child_index == 2
    assert np.array_equal(result.state, r.child(2))


@pytest.mark.parametrize("policy", [lambda d,k: [0.]*k, lambda d,k: [-1.]*k,
                                     lambda d,k: [float("nan")]*k, lambda d,k: [1.]])
def test_invalid_weight_policy_rejected(policy):
    with pytest.raises(ValueError):
        RankSelectionCoupling(rank_policy=policy)(request())


def test_complement_modes_reuse_the_specified_child():
    class ControlledRng:
        def __init__(self):
            self.draws = iter([.5, .999, .5])
        def random(self):
            return next(self.draws)
    children = np.array([[-5., 10., 0.], [-1., 20., 0.], [-3., 30., 0.]])
    for mode, expected in [("first", 10.), ("nearest_projection", 20.)]:
        r = replace(request(delta=2., k=3), children=children, rng=ControlledRng())
        out = RankSelectionCoupling(rank_policy="max", residual_complement=mode)(r)
        assert not out.accepted
        assert out.state[1] == expected


def test_callable_rank_policy_is_normalized_and_target_distributed():
    policy = lambda delta, k: [3., 1., 2., 4.]
    assert rank_weights(.7, 4, policy) == (.3, .1, .2, .4)
    report = check_exactness(RankSelectionCoupling(rank_policy=policy), delta=.7,
                             num_children=4, num_samples=1200, alpha=.0001, seed=97)
    assert report.passed, report


def test_bad_configuration_and_zero_noise_rejected():
    with pytest.raises(TypeError):
        RankSelectionCoupling(temperature=2.)
    with pytest.raises(ValueError):
        RankSelectionCoupling(residual_complement="nearest_vector")
    with pytest.raises(ValueError):
        RankSelectionCoupling(residual_method="quadrature")
    with pytest.raises(ValueError):
        RankSelectionCoupling(residual_method="rejection", residual_max_trials=0)
    with pytest.raises(ValueError):
        RankSelectionCoupling()(replace(request(), sigma=0.))


class AffineTarget(TargetTransition):
    def means(self, indices_in_batch, states, steps):
        return .6 * states + .25
    def freeze_drift(self, states, means, steps):
        return means - states
    def apply_drift(self, drift, states, steps):
        return states + drift


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_batched_trajectory_law_with_picard_and_truncation(iterations):
    sampler = BatchedSpeculativeSampler(
        target=AffineTarget(), proposal=IdentityProposal(),
        schedule=ConstantSchedule(.7), tree=DraftTree.from_widths([2, 1, 3]),
        verifier=RankSelectionCoupling(), num_steps=5,
        proposal_refinement_iters=iterations, check_contract=True,
        keep_trajectories=True,
    )
    result = sampler.sample(np.zeros((160, 2)), rng=np.random.default_rng(17+iterations))
    x = result.trajectories
    residuals = (x[:, 1:] - .6*x[:, :-1] - .25) / .7
    assert residuals.shape == (160, 5, 2)
    assert kstest(residuals.ravel(), "norm").statistic < .055
    assert abs(np.mean(residuals[:, 1:] * residuals[:, :-1])) < .12


def test_scalar_and_single_row_batch_agree():
    def kwargs():
        return dict(target=AffineTarget(), proposal=IdentityProposal(),
                    schedule=ConstantSchedule(.7), tree=DraftTree.uniform(2,3),
                    verifier=RankSelectionCoupling(), num_steps=7,
                    proposal_refinement_iters=1, check_contract=True)
    scalar = SpeculativeSampler(**kwargs()).sample(np.zeros(2), rng=np.random.default_rng(37))
    batch = BatchedSpeculativeSampler(**kwargs(), keep_trajectories=True).sample(
        np.zeros((1,2)), rng=np.random.default_rng(37))
    np.testing.assert_array_equal(scalar.trajectory, batch.trajectories[0])


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_cpu_exactness(dtype_name):
    torch = pytest.importorskip("torch")
    ref = torch.zeros(4, dtype=getattr(torch, dtype_name), device="cpu")
    report = check_exactness(RankSelectionCoupling(), delta=1.5, num_children=3,
                             num_samples=1200, alpha=.0001, seed=9, array_like=ref)
    assert report.passed, report
