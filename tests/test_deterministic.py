"""Deterministic transitions, exact root reuse and logical NFE accounting."""
import numpy as np
import pytest

from specdiff import (
    BatchedSpeculativeSampler, SpeculativeSampler, TargetTransition,
    DelayedDriftProposal, DraftTree, TabulatedSchedule, ConstantSchedule,
    create_verifier, VerifyRequest,
)
from specdiff.kernels import ProposalTransition
from specdiff.verify import verify_transition


class BoundedTarget(TargetTransition):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.seen = []

    def means(self, indices_in_batch, states, steps):
        assert all(0 <= n < self.horizon for n in steps)
        self.seen.extend((i, n, x.copy()) for i, n, x in zip(indices_in_batch, steps, states))
        return 0.5 * states + 0.25

    def freeze_drift(self, states, means, steps):
        return means - states

    def apply_drift(self, drift, states, steps):
        return states + drift


class ExactProposal(ProposalTransition):
    def means(self, indices_in_batch, states, steps):
        return 0.5 * states + 0.25


@pytest.mark.parametrize("rule", ["rmc", "d-grs", "paws"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("scales", [[0.] * 5, [0., .2, 0., .2, 0.]])
def test_full_trajectory_and_no_terminal_target_call(rule, batched, scales):
    target = BoundedTarget(len(scales))
    cls = BatchedSpeculativeSampler if batched else SpeculativeSampler
    kwargs = {"keep_trajectories": True} if batched else {}
    sampler = cls(
        target, ExactProposal(), TabulatedSchedule(scales),
        DraftTree.uniform(1 if rule == "rmc" else 2, 3), create_verifier(rule),
        num_steps=len(scales), evaluate_leaves=True, check_contract=True, **kwargs,
    )
    init = np.array([[.1, .3], [.5, .7]]) if batched else np.array([.1, .3])
    result = sampler.sample(init, rng=np.random.default_rng(71))
    paths = result.trajectories if batched else result.trajectory[None]
    assert np.array_equal(paths[:, 0], init if batched else init[None])
    for n, sigma in enumerate(scales):
        if sigma == 0:
            assert np.array_equal(paths[:, n + 1], .5 * paths[:, n] + .25)
    assert all(n < len(scales) for _, n, _ in target.seen)
    assert result.target_calls == 2
    if batched:
        assert result.target_calls_per_trajectory == (2, 2)
        assert sum(result.target_states_per_trajectory) == result.target_states_evaluated


@pytest.mark.parametrize("difference", [0., 1e-7])
def test_dirac_acceptance_requires_exact_equality(difference):
    mean = np.array([1.])
    request = VerifyRequest(step=0, proposal_mean=mean + difference,
                            target_mean=mean, sigma=0., children=np.array([[1. + difference]]))
    result = verify_transition(create_verifier("d-grs"), request)
    assert np.array_equal(result.state, mean)
    assert result.accepted == (difference == 0.)


@pytest.mark.parametrize("batched", [False, True])
def test_initialization_mean_is_reused_even_if_drift_roundtrip_rounds(batched):
    class RoundedTarget(BoundedTarget):
        def freeze_drift(self, states, means, steps):
            # Mimic a loss of precision in the inverse/reapply calculation.
            return super().freeze_drift(states, means, steps) + 1e-7

    target = RoundedTarget(1)
    cls = BatchedSpeculativeSampler if batched else SpeculativeSampler
    sampler = cls(target, DelayedDriftProposal(target), ConstantSchedule(0.),
                  DraftTree.chain(1), create_verifier("rmc"), num_steps=1,
                  evaluate_leaves=True, check_contract=True)
    init = np.array([[.1], [.2]]) if batched else np.array([.1])
    for _ in range(2):  # Resetting a sampler must not carry evaluations into another run.
        result = sampler.sample(init, rng=np.random.default_rng(7))
        assert result.target_calls == 1
        assert result.rounds[0].proposal_target_calls == 1
        assert result.rounds[0].verification_target_calls == 0
        assert result.acceptance_rate == 1.
        if batched:
            assert result.target_calls_per_trajectory == (1, 1)
            assert result.target_states_per_trajectory == (1, 1)


def test_per_image_calls_count_batches_not_nodes_without_round_records():
    target = BoundedTarget(5)
    target([0, 0, 1], np.zeros((3, 2)), [0, 1, 0])
    target([1], np.zeros((1, 2)), [1])
    assert target.num_calls == 2
    assert dict(target.calls_per_image) == {0: 1, 1: 2}
    assert dict(target.states_per_image) == {0: 2, 1: 2}
    sampler = BatchedSpeculativeSampler(
        target, DelayedDriftProposal(target), ConstantSchedule(.2),
        DraftTree.uniform(2, 2), create_verifier("d-grs"), num_steps=5,
    )
    result = sampler.sample(np.zeros((3, 2)), rng=np.random.default_rng(8), record=False)
    assert result.rounds == ()
    assert all(c > 0 for c in result.target_calls_per_trajectory)
    assert sum(result.target_states_per_trajectory) == result.target_states_evaluated
    assert result.mean_isolated_speedup == pytest.approx(
        np.mean([5 / c for c in result.target_calls_per_trajectory]))


@pytest.mark.parametrize("scale", [-.1, float("nan"), float("inf")])
def test_invalid_noise_scales_rejected(scale):
    with pytest.raises(ValueError, match="finite non-negative"):
        ConstantSchedule(scale)(0)
