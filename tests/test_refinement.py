"""Picard refinement recurrence, cache, API, and sampler-integration tests."""

import inspect

import numpy as np
import pytest

from specdiff import (
    BatchedSpeculativeSampler,
    ConstantSchedule,
    DraftTree,
    DelayedDriftProposal,
    IdentityProposal,
    RefinementUpdate,
    ReflectionMaximalCoupling,
    SpeculativeSampler,
    TargetTransition,
    Verifier,
    VerifyResult,
)


class AffineTarget(TargetTransition):
    def __init__(self, a=0.6, shift=0.25):
        super().__init__()
        self.a = float(a)
        self.shift = float(shift)

    def means(self, indices_in_batch, states, steps):
        shape = (len(steps),) + (1,) * (states.ndim - 1)
        offsets = np.asarray(
            [self.shift * (step + 1) + 0.01 * index
             for index, step in zip(indices_in_batch, steps)],
            dtype=states.dtype,
        ).reshape(shape)
        return self.a * states + offsets


class AcceptFirst(Verifier):
    max_children = None

    def __init__(self):
        self.requests = []

    def reset(self):
        self.requests = []

    def verify(self, request):
        self.requests.append(request)
        return VerifyResult(
            state=request.child(0),
            accepted=True,
            child_index=0,
            proposals_examined=1,
        )


def scalar_sampler(*, target=None, tree=None, iterations=None, callback=None, **kwargs):
    target = target or AffineTarget()
    tree = tree or DraftTree.chain(3)
    return target, SpeculativeSampler(
        target=target,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        tree=tree,
        verifier=AcceptFirst(),
        num_steps=tree.depth,
        proposal_refinement_iters=iterations,
        refinement_update_fn=callback,
        **kwargs,
    )


def test_constructor_semantics_and_scalar_batched_defaults_match():
    scalar = inspect.signature(SpeculativeSampler.__init__).parameters
    batched = inspect.signature(BatchedSpeculativeSampler.__init__).parameters
    for name in ("proposal_refinement_iters", "refinement_update_fn"):
        assert scalar[name].default == batched[name].default

    for bad in (-1, True, 1.5, "2"):
        with pytest.raises((TypeError, ValueError)):
            scalar_sampler(iterations=bad)

    _, sampler = scalar_sampler(iterations=np.int64(2))
    assert sampler.proposal_refinement_iters == 2


def test_zero_refinement_is_bitwise_identical_and_cost_neutral():
    common = dict(tree=DraftTree.uniform(2, 2), num_steps=6)
    target_a, target_b = AffineTarget(), AffineTarget()
    verifier_a, verifier_b = AcceptFirst(), AcceptFirst()
    sampler_a = SpeculativeSampler(
        target=target_a,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        verifier=verifier_a,
        **common,
    )
    sampler_b = SpeculativeSampler(
        target=target_b,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        verifier=verifier_b,
        proposal_refinement_iters=0,
        **common,
    )
    result_a = sampler_a.sample(np.zeros(2), rng=np.random.default_rng(4))
    result_b = sampler_b.sample(np.zeros(2), rng=np.random.default_rng(4))
    assert np.array_equal(result_a.trajectory, result_b.trajectory)
    assert result_a.target_calls == result_b.target_calls
    assert result_a.target_states_evaluated == result_b.target_states_evaluated
    assert all(record.refinement_iters == 0 for record in result_b.rounds)


def test_chain_picard_recurrence_and_cache_accounting():
    target, sampler = scalar_sampler(iterations=2)
    root = np.asarray([0.3, -0.4])
    sigma = 0.2
    seed = 7

    rng = np.random.default_rng(seed)
    innovations = sigma * rng.standard_normal((3, 2))
    old = [root.copy()]
    for edge in innovations:
        old.append(old[-1] + edge)
    for _ in range(2):
        new = [root.copy()]
        for depth, edge in enumerate(innovations):
            target_mean = target.a * old[depth] + target.shift * (depth + 1)
            increment = target_mean - old[depth]
            new.append(new[depth] + increment + edge)
        old = new

    result = sampler.sample(root, rng=np.random.default_rng(seed))
    assert np.allclose(result.trajectory, np.stack(old))
    record = result.rounds[0]
    assert record.refinement_target_calls == 2
    assert record.refinement_target_states_evaluated == 6
    assert record.verification_target_calls == 1
    assert record.verification_target_states_evaluated == 1
    assert record.verification_target_means_reused == 2
    assert record.target_calls == 3


def test_full_depth_refinement_avoids_final_target_call():
    _, sampler = scalar_sampler(tree=DraftTree.chain(2), iterations=2)
    result = sampler.sample(np.zeros(2), rng=np.random.default_rng(1))
    record = result.rounds[0]
    assert record.refinement_target_calls == 2
    assert record.verification_target_calls == 0
    assert record.verification_target_states_evaluated == 0
    assert record.verification_target_means_reused == 2
    assert result.target_calls == 2


def test_evaluate_leaves_still_needs_the_leaf_target_mean():
    _, sampler = scalar_sampler(
        tree=DraftTree.chain(2), iterations=2, evaluate_leaves=True
    )
    result = sampler.sample(np.zeros(2), rng=np.random.default_rng(1))
    record = result.rounds[0]
    assert record.verification_target_calls == 1
    assert record.verification_target_states_evaluated == 1
    assert record.verification_target_means_reused == 2


def test_irregular_tree_callback_receives_row_local_metadata():
    tree = DraftTree((-1, 0, 0, 1, 2, 2))
    seen = []

    def update(request):
        seen.append(request)
        return RefinementUpdate(np.zeros_like(request.parent_states))

    _, sampler = scalar_sampler(tree=tree, iterations=1, callback=update)
    result = sampler.sample(np.zeros(1), rng=np.random.default_rng(3))
    assert len(seen) == 1
    request = seen[0]
    assert request.nodes == (0, 1, 2)
    assert request.steps == (0, 1, 1)
    assert request.indices_in_batch == (0, 0, 0)
    assert result.rounds[0].verification_target_means_reused == 0
    assert result.rounds[0].verified == 3


@pytest.mark.parametrize("kind", ["raw", "shape", "nonfinite"])
def test_malformed_callback_output_is_rejected(kind):
    def update(request):
        if kind == "raw":
            return np.zeros_like(request.parent_states)
        if kind == "shape":
            return RefinementUpdate(np.zeros_like(request.parent_states[:1]))
        values = np.zeros_like(request.parent_states)
        values[0] = np.nan
        return RefinementUpdate(values)

    _, sampler = scalar_sampler(iterations=1, callback=update)
    with pytest.raises((TypeError, ValueError)):
        sampler.sample(np.zeros(2), rng=np.random.default_rng(0))


def test_false_exact_target_cache_is_detected_in_contract_mode():
    def update(request):
        zeros = np.zeros_like(request.parent_states)
        return RefinementUpdate(increments=zeros, exact_target_means=zeros)

    _, sampler = scalar_sampler(
        tree=DraftTree.chain(1),
        iterations=1,
        callback=update,
        check_contract=True,
    )
    with pytest.raises(ValueError, match="reused target mean"):
        sampler.sample(np.ones(1), rng=np.random.default_rng(0))


def test_batch_of_one_matches_scalar_with_refinement():
    common = dict(
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.uniform(2, 3),
        num_steps=7,
        proposal_refinement_iters=2,
    )
    scalar_target, batch_target = AffineTarget(), AffineTarget()
    scalar = SpeculativeSampler(
        target=scalar_target,
        proposal=IdentityProposal(),
        verifier=AcceptFirst(),
        **common,
    )
    batched = BatchedSpeculativeSampler(
        target=batch_target,
        proposal=IdentityProposal(),
        verifier=AcceptFirst(),
        keep_trajectories=True,
        **common,
    )
    result_s = scalar.sample(np.zeros(2), rng=np.random.default_rng(11))
    result_b = batched.sample(np.zeros((1, 2)), rng=np.random.default_rng(11))
    assert np.array_equal(result_s.trajectory, result_b.trajectories[0])
    assert result_s.target_calls == result_b.target_calls
    assert result_s.target_states_evaluated == result_b.target_states_evaluated


def test_batched_final_call_exists_if_any_live_lookahead_exceeds_j():
    target = AffineTarget()
    sampler = BatchedSpeculativeSampler(
        target=target,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.chain(3),
        verifier=AcceptFirst(),
        num_steps=2,
        proposal_refinement_iters=1,
    )
    result = sampler.sample(np.zeros((3, 1)), rng=np.random.default_rng(2))
    record = result.rounds[0]
    assert record.refinement_target_calls == 1
    assert record.verification_target_calls == 1
    assert record.verification_target_means_reused == 3
    assert record.target_calls == 2

@pytest.mark.parametrize("prefetch", ["none", "nearest"])
def test_proposal_owned_target_accounting_is_complete(prefetch):
    target = AffineTarget()
    sampler = SpeculativeSampler(
        target=target,
        proposal=DelayedDriftProposal(target),
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.chain(1),
        verifier=AcceptFirst(),
        num_steps=2,
        prefetch=prefetch,
        proposal_refinement_iters=1,
    )
    result = sampler.sample(np.zeros(1), rng=np.random.default_rng(9))
    assert sum(record.target_calls for record in result.rounds) == result.target_calls
    assert (
        sum(record.target_states_evaluated for record in result.rounds)
        == result.target_states_evaluated
    )
    for record in result.rounds:
        assert record.target_calls == (
            record.proposal_target_calls
            + record.refinement_target_calls
            + record.verification_target_calls
        )
    if prefetch == "none":
        assert [record.proposal_target_calls for record in result.rounds] == [1, 1]
    else:
        assert [record.proposal_target_calls for record in result.rounds] == [1, 0]


def test_torch_refinement_and_exact_cache_reuse():
    torch = pytest.importorskip("torch")

    class TorchAffineTarget(TargetTransition):
        def means(self, indices_in_batch, states, steps):
            offsets = torch.as_tensor(
                [0.1 * (step + 1) for step in steps],
                dtype=states.dtype,
                device=states.device,
            ).reshape((len(steps),) + (1,) * (states.ndim - 1))
            return 0.7 * states + offsets

    target = TorchAffineTarget()
    sampler = SpeculativeSampler(
        target=target,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.chain(2),
        verifier=AcceptFirst(),
        num_steps=2,
        proposal_refinement_iters=2,
    )
    generator = torch.Generator().manual_seed(3)
    result = sampler.sample(torch.zeros(2), rng=generator)
    assert result.trajectory.shape == (3, 2)
    assert result.rounds[0].verification_target_calls == 0
    assert result.rounds[0].verification_target_means_reused == 2


def test_callback_rows_are_partition_invariant_for_picard_update():
    target, sampler = scalar_sampler(tree=DraftTree.uniform(2, 2), iterations=1)
    layout = sampler._refinement_layout(sampler.tree, 0)
    states = np.arange(sampler.tree.size, dtype=float).reshape((-1, 1))
    parent_states = states[list(layout.internal_ids)]
    full = target.means(layout.indices_in_batch, parent_states, layout.steps) - parent_states
    pieces = []
    for i in range(len(parent_states)):
        pieces.append(
            target.means(
                (layout.indices_in_batch[i],),
                parent_states[i:i + 1],
                (layout.steps[i],),
            ) - parent_states[i:i + 1]
        )
    assert np.array_equal(full, np.concatenate(pieces, axis=0))

def test_mixed_batched_lookaheads_use_one_final_call_if_any_row_needs_it():
    class StaggeredVerifier(Verifier):
        max_children = 1

        def verify(self, request):
            if request.index_in_batch == 0:
                return VerifyResult(
                    request.child(0), accepted=True, child_index=0
                )
            return VerifyResult(request.target_mean, accepted=False)

    target = AffineTarget()
    sampler = BatchedSpeculativeSampler(
        target=target,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.chain(3),
        verifier=StaggeredVerifier(),
        num_steps=5,
        proposal_refinement_iters=1,
    )
    result = sampler.sample(np.zeros((2, 1)), rng=np.random.default_rng(12))
    mixed = next(record for record in result.rounds if record.start_steps == (3, 1))
    assert mixed.verification_target_calls == 1
    for record in result.rounds:
        needs_final = any(
            1 < min(sampler.tree.depth, sampler.num_steps - start)
            for start in record.start_steps
        )
        assert record.verification_target_calls == int(needs_final)


def test_refined_rmc_preserves_affine_target_law():
    a, shift, sigma, num_steps = 0.7, 0.15, 0.3, 3
    target = AffineTarget(a=a, shift=shift)
    sampler = SpeculativeSampler(
        target=target,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(sigma),
        tree=DraftTree.chain(3),
        verifier=ReflectionMaximalCoupling(),
        num_steps=num_steps,
        proposal_refinement_iters=1,
    )
    rng = np.random.default_rng(21)
    samples = np.asarray([
        sampler.sample(np.zeros(1), rng=rng).sample[0] for _ in range(2500)
    ])

    mean, variance = 0.0, 0.0
    for step in range(num_steps):
        mean = a * mean + shift * (step + 1)
        variance = a * a * variance + sigma * sigma
    standard_error = np.sqrt(variance / len(samples))
    assert abs(samples.mean() - mean) < 5 * standard_error
    assert abs(samples.var() / variance - 1.0) < 0.1
