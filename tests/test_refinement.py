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
    ProposalTransition,
    RefinementUpdate,
    ReflectionMaximalCoupling,
    SpeculativeSampler,
    TargetTransition,
    Verifier,
    VerifyResult,
    picard_drift_update_fn,
    picard_jtx_update_fn,
    picard_update_fn,
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

    # A translation-like toy map: the increment is the thing to freeze.
    def freeze_drift(self, states, means, steps):
        return means - states

    def apply_drift(self, drift, states, steps):
        return states + drift


class VelocitySplitTarget(TargetTransition):
    """``m = a x + b v(x)`` that freezes ``v``, the shape of the churn kernels."""

    def __init__(self, a=0.8, b=-0.5, slope=0.5, bend=1.0):
        super().__init__()
        self.a, self.b = float(a), float(b)
        self.slope, self.bend = float(slope), float(bend)

    def velocity(self, states, steps):
        shape = (len(steps),) + (1,) * (states.ndim - 1)
        offsets = np.asarray([0.2 * (step + 1) for step in steps], dtype=states.dtype)
        return (
            self.slope * states + self.bend * np.tanh(states) + offsets.reshape(shape)
        )

    def means(self, indices_in_batch, states, steps):
        return self.a * states + self.b * self.velocity(states, steps)

    def freeze_drift(self, states, means, steps):
        return (means - self.a * states) / self.b

    def apply_drift(self, drift, states, steps):
        return self.a * states + self.b * drift


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


def scalar_sampler(*, target=None, proposal=None, tree=None, iterations=None, callback=None, **kwargs):
    target = target or AffineTarget()
    tree = tree or DraftTree.chain(3)
    return target, SpeculativeSampler(
        target=target,
        proposal=proposal or IdentityProposal(),
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


@pytest.mark.parametrize(
    "target_type, callback",
    [(AffineTarget, None), (VelocitySplitTarget, picard_drift_update_fn),
     (VelocitySplitTarget, picard_jtx_update_fn)],
)
def test_batch_of_one_matches_scalar_with_refinement(target_type, callback):
    common = dict(
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.uniform(2, 3),
        num_steps=7,
        proposal_refinement_iters=2,
        refinement_update_fn=callback,
    )
    scalar_target, batch_target = target_type(), target_type()
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


@pytest.mark.parametrize("callback", [None, picard_drift_update_fn, picard_jtx_update_fn])
def test_torch_refinement_and_exact_cache_reuse(callback):
    torch = pytest.importorskip("torch")

    class TorchAffineTarget(TargetTransition):
        def means(self, indices_in_batch, states, steps):
            offsets = torch.as_tensor(
                [0.1 * (step + 1) for step in steps],
                dtype=states.dtype,
                device=states.device,
            ).reshape((len(steps),) + (1,) * (states.ndim - 1))
            return 0.7 * states + offsets

        def freeze_drift(self, states, means, steps):
            return means - states

        def apply_drift(self, drift, states, steps):
            return states + drift

    target = TorchAffineTarget()
    sampler = SpeculativeSampler(
        target=target,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.chain(2),
        verifier=AcceptFirst(),
        num_steps=2,
        proposal_refinement_iters=2,
        refinement_update_fn=callback,
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


@pytest.mark.parametrize("callback", [None, picard_jtx_update_fn])
def test_refined_rmc_preserves_affine_target_law(callback):
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
        refinement_update_fn=callback,
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


def test_drift_picard_recurrence_and_cache_accounting():
    target = VelocitySplitTarget()
    _, sampler = scalar_sampler(
        target=target, iterations=2, callback=picard_drift_update_fn
    )
    root = np.asarray([0.3, -0.4])
    innovations = 0.2 * np.random.default_rng(7).standard_normal((3, 2))
    old = [root.copy()]
    for edge in innovations:
        old.append(old[-1] + edge)
    for _ in range(2):
        # The affine part sees the rebuilt parent; only v comes from the snapshot.
        new = [root.copy()]
        for depth, edge in enumerate(innovations):
            frozen = target.velocity(old[depth][None], (depth,))[0]
            new.append(target.a * new[depth] + target.b * frozen + edge)
        old = new

    result = sampler.sample(root, rng=np.random.default_rng(7))
    assert np.allclose(result.trajectory, np.stack(old))
    record = result.rounds[0]
    assert record.refinement_target_calls == 2
    assert record.refinement_target_states_evaluated == 6
    assert record.verification_target_calls == 1
    assert record.verification_target_means_reused == 2


def test_full_depth_drift_refinement_reaches_the_exact_target_chain():
    target = VelocitySplitTarget()
    _, sampler = scalar_sampler(
        target=target, iterations=3, callback=picard_drift_update_fn
    )
    root = np.asarray([0.3, -0.4])
    innovations = 0.2 * np.random.default_rng(5).standard_normal((3, 2))
    exact = [root]
    for depth, edge in enumerate(innovations):
        exact.append(target.means((0,), exact[-1][None], (depth,))[0] + edge)

    result = sampler.sample(root, rng=np.random.default_rng(5))
    assert np.allclose(result.trajectory, np.stack(exact))
    record = result.rounds[0]
    assert record.verification_target_calls == 0
    assert record.verification_target_means_reused == 3


def test_drift_update_matches_increment_update_for_a_translation_drift():
    # AffineTarget freezes the increment itself, so the updates agree up to rounding.
    results = []
    for callback in (picard_update_fn, picard_drift_update_fn):
        _, sampler = scalar_sampler(
            tree=DraftTree.uniform(2, 3), iterations=2, callback=callback
        )
        results.append(sampler.sample(np.zeros(2), rng=np.random.default_rng(4)))
    assert np.allclose(results[0].trajectory, results[1].trajectory)
    assert results[0].target_calls == results[1].target_calls


def test_frozen_drift_is_the_default_refinement_update():
    _, scalar = scalar_sampler(iterations=1)
    batched = BatchedSpeculativeSampler(
        target=AffineTarget(),
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.chain(2),
        verifier=AcceptFirst(),
        num_steps=2,
        proposal_refinement_iters=1,
    )
    assert scalar.refinement_update_fn is picard_drift_update_fn
    assert batched.refinement_update_fn is picard_drift_update_fn


def test_converged_drift_parents_keep_the_exact_target_mean():
    verifier = AcceptFirst()
    sampler = SpeculativeSampler(
        target=VelocitySplitTarget(),
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2),
        tree=DraftTree.chain(3),
        verifier=verifier,
        num_steps=3,
        proposal_refinement_iters=2,
    )
    sampler.sample(np.asarray([0.3, -0.4]), rng=np.random.default_rng(8))
    # Parents above depth J = 2 have converged: their proposal mean must be
    # the target mean bit for bit, not apply(freeze(m)) up to rounding.
    for request in verifier.requests[:2]:
        assert np.array_equal(request.proposal_mean, request.target_mean)
    assert not np.array_equal(
        verifier.requests[2].proposal_mean, verifier.requests[2].target_mean
    )


def test_refinement_update_needs_exactly_one_of_increments_and_drifts():
    zeros = np.zeros((2, 1))
    with pytest.raises(TypeError):
        RefinementUpdate()
    with pytest.raises(TypeError):
        RefinementUpdate(increments=zeros, drifts=zeros)
    assert RefinementUpdate(drifts=zeros).increments is None


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "apply_drift"])
def test_malformed_drift_refinement_is_rejected(kind):
    class BadApply(VelocitySplitTarget):
        def apply_drift(self, drift, states, steps):
            return super().apply_drift(drift, states, steps)[..., :1]

    def update(request):
        values = np.zeros_like(request.parent_states)
        if kind == "shape":
            values = values[:1]
        elif kind == "nonfinite":
            values[0] = np.nan
        return RefinementUpdate(drifts=values)

    target = BadApply() if kind == "apply_drift" else VelocitySplitTarget()
    _, sampler = scalar_sampler(target=target, iterations=1, callback=update)
    with pytest.raises(ValueError):
        sampler.sample(np.zeros(2), rng=np.random.default_rng(0))


def test_refined_rmc_with_drift_update_preserves_target_law():
    target = VelocitySplitTarget(a=0.8, b=-0.5, slope=0.5, bend=0.0)
    sigma, num_steps = 0.3, 3
    sampler = SpeculativeSampler(
        target=target,
        proposal=IdentityProposal(),
        schedule=ConstantSchedule(sigma),
        tree=DraftTree.chain(3),
        verifier=ReflectionMaximalCoupling(),
        num_steps=num_steps,
        proposal_refinement_iters=1,
        refinement_update_fn=picard_drift_update_fn,
    )
    rng = np.random.default_rng(21)
    samples = np.asarray([
        sampler.sample(np.zeros(1), rng=rng).sample[0] for _ in range(2500)
    ])

    # bend = 0 makes the target linear: x -> (a + b slope) x + 0.2 b (step + 1).
    gain = target.a + target.b * target.slope
    mean, variance = 0.0, 0.0
    for step in range(num_steps):
        mean = gain * mean + 0.2 * target.b * (step + 1)
        variance = gain * gain * variance + sigma * sigma
    standard_error = np.sqrt(variance / len(samples))
    assert abs(samples.mean() - mean) < 5 * standard_error
    assert abs(samples.var() / variance - 1.0) < 0.1


class NonlinearBaseProposal(ProposalTransition):
    """A row-local draft whose nonlinear part must see the rebuilt parent."""

    def means(self, indices_in_batch, states, steps):
        offsets = np.asarray([
            0.07 * (step + 1) + 0.03 * index
            for index, step in zip(indices_in_batch, steps)
        ], dtype=states.dtype)
        shape = (len(steps),) + (1,) * (states.ndim - 1)
        return 0.4 * states + 0.2 * np.sin(states) + offsets.reshape(shape)


@pytest.mark.parametrize("iterations", [0, 1, 2, 4, 6])
def test_jtx_matches_pdf_recurrence_on_every_edge(iterations):
    from dataclasses import replace
    from specdiff.ops import resolve_backend
    from specdiff.refinement import refine_tree, reusable_target_rows

    # Uneven sibling counts and depth four, with image-shaped states.
    tree = DraftTree((-1, 0, 0, 1, 2, 2, 3, 6))
    target = VelocitySplitTarget()
    proposal = NonlinearBaseProposal()
    _, sampler = scalar_sampler(target=target, tree=tree)
    layout = sampler._refinement_layout(tree, 3)
    layout = replace(layout, indices_in_batch=(4,) * len(layout.internal_ids))
    root = np.asarray([[0.3, -0.4], [0.6, 0.2]])
    ops = resolve_backend(root)
    noise = 0.2 * np.random.default_rng(17).standard_normal((tree.size, *root.shape))
    noise[0] = 0
    states = np.zeros_like(noise)
    states[0] = root
    means = np.zeros_like(noise)

    def base(x, step):
        return proposal.means((4,), x[None], (step,))[0]

    def exact(x, step):
        return target.means((4,), x[None], (step,))[0]

    # Initial proposal, eq. (2).
    for level in layout.levels:
        for u, pos in zip(level.parent_ids, level.parent_positions):
            means[u] = base(states[u], layout.steps[pos])
            for v in tree.children(u):
                states[v] = means[u] + noise[v]
    expected = states.copy()
    expected_means = means.copy()
    for _ in range(iterations):
        old = expected.copy()
        # Direct p(new) + [q(old) - p(old)], eqs. (3)-(4).
        # In particular, do not subtract the previous corrected mean.
        for level in layout.levels:
            for u, pos in zip(level.parent_ids, level.parent_positions):
                step = layout.steps[pos]
                error = exact(old[u], step) - base(old[u], step)
                expected_means[u] = base(expected[u], step) + error
                for v in tree.children(u):
                    expected[v] = expected_means[u] + noise[v]

    original_noise = noise.copy()
    cache = refine_tree(
        states=states, proposal_means=means, scaled_innovations=noise,
        layout=layout, iterations=iterations, update_fn=picard_jtx_update_fn,
        target=target, proposal=proposal, ops=ops,
    )
    np.testing.assert_allclose(states, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(means, expected_means, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(noise, original_noise)
    np.testing.assert_array_equal(states[0], root)
    assert target.num_calls == iterations
    assert target.num_states == iterations * len(layout.internal_ids)
    reused = reusable_target_rows(cache, target=target, final_states=states, ops=ops)
    if iterations == 0:
        assert cache is None
    elif iterations >= tree.depth:
        assert len(reused) == len(layout.internal_ids)
        for u, step in zip(layout.internal_ids, layout.steps):
            np.testing.assert_array_equal(means[u], exact(states[u], step))
    else:
        assert 0 < len(reused) < len(layout.internal_ids)


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_jtx_verifier_receives_corrected_means_and_final_target(iterations):
    target = VelocitySplitTarget()
    verifier = AcceptFirst()
    _, sampler = scalar_sampler(
        target=target, proposal=NonlinearBaseProposal(), iterations=iterations,
        callback=picard_jtx_update_fn, check_contract=True,
    )
    sampler.verifier = verifier
    root = np.asarray([0.3, -0.4])
    result = sampler.sample(root, rng=np.random.default_rng(7))
    noise = 0.2 * np.random.default_rng(7).standard_normal((3, 2))
    for depth, request in enumerate(verifier.requests):
        np.testing.assert_allclose(request.child(0), request.proposal_mean + noise[depth])
        expected_target = target.means((0,), result.trajectory[depth][None], (depth,))[0]
        np.testing.assert_array_equal(request.target_mean, expected_target)
        if depth < iterations:
            np.testing.assert_array_equal(request.proposal_mean, request.target_mean)
    record = result.rounds[0]
    assert record.refinement_target_calls == iterations
    assert record.verification_target_means_reused == iterations
    assert record.verification_target_states_evaluated == 3 - iterations


@pytest.mark.parametrize("prefetch", ["none", "nearest"])
def test_jtx_delayed_drift_matches_frozen_drift_across_rounds(prefetch):
    results = []
    for callback in (picard_drift_update_fn, picard_jtx_update_fn):
        target = VelocitySplitTarget()
        sampler = BatchedSpeculativeSampler(
            target=target, proposal=DelayedDriftProposal(target),
            schedule=ConstantSchedule(0.2), tree=DraftTree.chain(3),
            verifier=ReflectionMaximalCoupling(), num_steps=7,
            proposal_refinement_iters=2, refinement_update_fn=callback,
            prefetch=prefetch, keep_trajectories=True, check_contract=True,
        )
        results.append(sampler.sample(np.zeros((3, 2)), rng=np.random.default_rng(19)))
    for a, b in zip(results[0].trajectories, results[1].trajectories):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)
    assert results[0].target_calls == results[1].target_calls
    assert results[0].target_states_evaluated == results[1].target_states_evaluated


def test_jtx_update_requires_proposal_and_exclusive_snapshot_means():
    from specdiff import RefinementRequest
    from specdiff.ops import resolve_backend

    zeros = np.zeros((2, 1))
    with pytest.raises(TypeError, match="exact_target_means"):
        RefinementUpdate(base_proposal_means=zeros)
    for field in ("increments", "drifts"):
        with pytest.raises(TypeError, match="exactly one"):
            RefinementUpdate(
                base_proposal_means=zeros, exact_target_means=zeros, **{field: zeros}
            )
    request = RefinementRequest(
        iteration=0, indices_in_batch=(0, 0), nodes=(0, 1), steps=(0, 1),
        parent_states=zeros, current_proposal_means=zeros, sigmas=(0.2, 0.2),
        target=AffineTarget(), backend=resolve_backend(zeros),
    )
    with pytest.raises(ValueError, match="base proposal"):
        picard_jtx_update_fn(request)


@pytest.mark.parametrize("kind", ["snapshot_shape", "snapshot_nonfinite", "rebuilt_shape", "rebuilt_nonfinite"])
def test_jtx_rejects_malformed_base_proposal_means(kind):
    class BadProposal(NonlinearBaseProposal):
        corrupt = False

        def means(self, indices_in_batch, states, steps):
            out = super().means(indices_in_batch, states, steps)
            if self.corrupt:
                if kind.endswith("shape"):
                    return out[..., :1]
                out[..., 0] = np.nan
            return out

    proposal = BadProposal()

    def update(request):
        proposal.corrupt = kind.startswith("snapshot")
        result = picard_jtx_update_fn(request)
        proposal.corrupt = kind.startswith("rebuilt")
        return result

    _, sampler = scalar_sampler(proposal=proposal, iterations=1, callback=update)
    with pytest.raises(ValueError, match="base proposal means"):
        sampler.sample(np.zeros(2), rng=np.random.default_rng(1))


@pytest.mark.parametrize("callback", [picard_update_fn, picard_drift_update_fn, picard_jtx_update_fn])
@pytest.mark.parametrize("batched", [False, True])
def test_refinement_sweeps_are_capped_by_actual_lookahead(callback, batched):
    target = VelocitySplitTarget()
    sampler_type = BatchedSpeculativeSampler if batched else SpeculativeSampler
    sampler = sampler_type(
        target=target, proposal=IdentityProposal(), schedule=ConstantSchedule(0.2),
        tree=DraftTree.chain(3), verifier=AcceptFirst(), num_steps=4,
        proposal_refinement_iters=5, refinement_update_fn=callback,
    )
    root = np.zeros((1, 2)) if batched else np.zeros(2)
    result = sampler.sample(root, rng=np.random.default_rng(42))
    assert sampler.proposal_refinement_iters == 5
    assert [r.refinement_iters for r in result.rounds] == [3, 1]
    assert [r.refinement_target_calls for r in result.rounds] == [3, 1]
    assert [r.refinement_target_states_evaluated for r in result.rounds] == [9, 1]
    assert result.target_calls == 4


def test_batched_sweep_cap_keeps_enough_iterations_for_longer_live_trees():
    class StaggeredVerifier(Verifier):
        max_children = 1

        def verify(self, request):
            if request.index_in_batch == 0:
                return VerifyResult(request.child(0), accepted=True, child_index=0)
            return VerifyResult(request.target_mean, accepted=False)

    sampler = BatchedSpeculativeSampler(
        target=VelocitySplitTarget(), proposal=IdentityProposal(),
        schedule=ConstantSchedule(0.2), tree=DraftTree.chain(3),
        verifier=StaggeredVerifier(), num_steps=5,
        proposal_refinement_iters=3, refinement_update_fn=picard_jtx_update_fn,
    )
    result = sampler.sample(np.zeros((2, 2)), rng=np.random.default_rng(42))
    assert result.rounds[1].start_steps == (3, 1)
    assert [r.refinement_iters for r in result.rounds] == [3, 3, 3, 2, 1]
    assert [r.refinement_target_calls for r in result.rounds] == [3, 3, 3, 2, 1]
