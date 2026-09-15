"""Broyden secants, row-local history, and exact-sampler integration."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from specdiff import (
    BatchedSpeculativeSampler, ConstantSchedule, DraftTree, IdentityProposal,
    RefinementRequest, RefinementUpdate, ReflectionMaximalCoupling,
    SpeculativeSampler, TargetTransition, Verifier, VerifyResult,
    picard_broyden_correction_update_fn as broyden, picard_jtx_update_fn,
)
from specdiff.ops import resolve_backend


class ToyTarget(TargetTransition):
    def __init__(self, nonlinear=0.1):
        super().__init__()
        self.nonlinear = nonlinear

    def means(self, indices_in_batch, states, steps):
        ops = resolve_backend(states)
        offsets = ops.scale_rows(states * 0 + 1, [
            0.2 + 0.01 * image + 0.02 * step
            for image, step in zip(indices_in_batch, steps)
        ])
        return 0.65 * states + self.nonlinear * states**2 + offsets

    def freeze_drift(self, states, means, steps):
        return means - states

    def apply_drift(self, drift, states, steps):
        return states + drift


class AcceptFirst(Verifier):
    max_children = None

    def verify(self, request):
        return VerifyResult(request.child(0), accepted=True, child_index=0)


def request(states, *, iteration=0, history=None, target=None, proposal=None):
    n = len(states)
    return RefinementRequest(
        iteration=iteration, indices_in_batch=tuple(range(n)), nodes=tuple(range(n)),
        steps=(2,) * n, parent_states=states, current_proposal_means=states,
        sigmas=(0.2,) * n, target=target or ToyTarget(),
        proposal=proposal or IdentityProposal(), backend=resolve_backend(states),
        history={} if history is None else history,
    )


def matrix(factors, dimension):
    return sum((np.outer(u, v) for u, v in factors), np.zeros((dimension, dimension)))


@pytest.mark.parametrize('memory', [1, 2, 3])
def test_factor_history_matches_dense_fifo_broyden_and_newest_secant(memory):
    rng = np.random.default_rng(8)
    req = request(rng.normal(size=(2, 2, 2)))
    dense_terms = [[], []]
    previous_states = previous_errors = None
    for iteration in range(6):
        states = rng.normal(size=(2, 2, 2))
        req = replace(req, iteration=iteration, parent_states=states)
        update = broyden(req, memory=memory)
        errors = update.exact_target_means - update.base_proposal_means
        for row, factors in enumerate(update.broyden_factors):
            if iteration:
                s = (states[row] - previous_states[row]).ravel()
                z = (errors[row] - previous_errors[row]).ravel()
                retained = dense_terms[row][-(memory - 1):] if memory > 1 else []
                B = sum(retained, np.zeros((4, 4)))
                dense_terms[row] = retained + [np.outer(z - B @ s, s) / (s @ s)]
                actual = matrix(factors, 4)
                np.testing.assert_allclose(actual, sum(dense_terms[row]), atol=1e-13)
                np.testing.assert_allclose(actual @ s, z, atol=1e-13)
            assert len(factors) == min(iteration, memory)
        previous_states, previous_errors = states.copy(), errors.copy()
    assert req.target.num_calls == 6
    assert req.target.num_states == 12


def test_history_is_partition_invariant_and_keyed_by_image_node_and_step():
    base = request(np.zeros((3, 2)))
    full_history, partitioned_history = {}, {}
    rng = np.random.default_rng(4)
    for iteration in range(4):
        states = rng.normal(size=(3, 2))
        full = broyden(replace(base, iteration=iteration, parent_states=states,
                               history=full_history))
        for row in [2, 0, 1]:
            one = replace(
                base, iteration=iteration, parent_states=states[row:row + 1],
                indices_in_batch=(base.indices_in_batch[row],), nodes=(base.nodes[row],),
                steps=(base.steps[row],), history=partitioned_history,
            )
            part = broyden(one)
            np.testing.assert_array_equal(matrix(part.broyden_factors[0], 2),
                                          matrix(full.broyden_factors[row], 2))
    fresh_step = broyden(replace(base, iteration=4, steps=(3,) * 3, history=full_history))
    assert fresh_step.broyden_factors == ((), (), ())
    reset = broyden(replace(base, history=full_history))
    assert reset.broyden_factors == ((), (), ())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_stationary_and_roundoff_sized_movements_do_not_create_factors(dtype):
    states = np.ones((2, 3), dtype=dtype)
    req = request(states)
    broyden(req)
    next_states = states.copy()
    next_states[1] += np.finfo(dtype).eps
    update = broyden(replace(req, iteration=1, parent_states=next_states))
    assert update.broyden_factors == ((), ())


@pytest.mark.parametrize('memory', [-1, True, 1.5, '2'])
def test_invalid_memory_is_rejected_before_target_evaluation(memory):
    req = request(np.zeros((1, 2)))
    with pytest.raises((ValueError, TypeError), match='memory'):
        broyden(req, memory=memory)
    assert req.target.num_calls == 0


@pytest.mark.parametrize('batched', [False, True])
def test_memory_zero_is_bitwise_jtx_and_history_does_not_leak_between_runs(batched):
    sampler_type = BatchedSpeculativeSampler if batched else SpeculativeSampler
    kwargs = dict(keep_trajectories=True) if batched else {}
    outputs = []
    for callback in [picard_jtx_update_fn, partial(broyden, memory=0), broyden]:
        sampler = sampler_type(
            target=ToyTarget(), proposal=IdentityProposal(),
            tree=DraftTree.uniform(2, 3), schedule=ConstantSchedule(0.2),
            num_steps=7, verifier=AcceptFirst(), proposal_refinement_iters=2,
            refinement_update_fn=callback, check_contract=True, **kwargs,
        )
        root = np.zeros((2, 2)) if batched else np.zeros(2)
        results = [sampler.sample(root, rng=np.random.default_rng(12)) for _ in range(2)]
        trajectories = [r.trajectories if batched else r.trajectory for r in results]
        np.testing.assert_array_equal(trajectories[0], trajectories[1])
        assert results[0].target_calls == results[1].target_calls
        outputs.append((trajectories[0], results[0].target_calls))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == outputs[1][1]
    assert not np.allclose(outputs[1][0], outputs[2][0])


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_two_sweeps_recover_linear_target_chain_using_secants(backend):
    torch = pytest.importorskip('torch') if backend == 'torch' else None
    outputs = []
    for callback in [picard_jtx_update_fn, broyden]:
        target = ToyTarget(nonlinear=0)
        seen = []

        def record(req):
            out = callback(req)
            seen.append(out)
            return out

        sampler = SpeculativeSampler(
            target=target, proposal=IdentityProposal(), tree=DraftTree.chain(5),
            schedule=ConstantSchedule(0.2), num_steps=5, verifier=AcceptFirst(),
            proposal_refinement_iters=2, refinement_update_fn=record,
            check_contract=True,
        )
        root = torch.tensor([0.4], dtype=torch.float64) if torch else np.array([0.4])
        rng = torch.Generator().manual_seed(11) if torch else np.random.default_rng(11)
        result = sampler.sample(root, rng=rng)
        noise_rng = torch.Generator().manual_seed(11) if torch else np.random.default_rng(11)
        noise = 0.2 * resolve_backend(root).randn_stack(5, root, noise_rng)
        exact = [root]
        for step in range(5):
            exact.append(target.means((0,), exact[-1][None], (step,))[0] + noise[step])
        exact = resolve_backend(root).stack_rows(exact)
        if callback is broyden:
            np.testing.assert_allclose(result.trajectory, exact, atol=1e-13)
            assert seen[0].broyden_factors == ((),) * 5
            assert any(seen[1].broyden_factors)
        outputs.append(np.asarray(result.trajectory))
        assert result.rounds[0].refinement_target_calls == 2
    assert not np.allclose(outputs[0], outputs[1])


def test_broyden_rmc_preserves_linear_target_moments():
    sampler = BatchedSpeculativeSampler(
        target=ToyTarget(nonlinear=0), proposal=IdentityProposal(),
        tree=DraftTree.chain(4), schedule=ConstantSchedule(0.3),
        num_steps=6, verifier=ReflectionMaximalCoupling(),
        proposal_refinement_iters=2, refinement_update_fn=broyden,
    )
    result = sampler.sample(np.zeros((1500, 2)), rng=np.random.default_rng(17))
    # Image conditioning adds 0.01 * image each step.
    mean = np.zeros(1500)
    variance = 0.0
    for step in range(6):
        mean = 0.65 * mean + 0.2 + 0.01 * np.arange(1500) + 0.02 * step
        variance = 0.65**2 * variance + 0.3**2
    residual = result.samples - mean[:, None]
    assert sum(sum(r.rejected) for r in result.rounds) > 0
    assert np.max(np.abs(residual.mean(axis=0))) < 5 * np.sqrt(variance / 1500)
    np.testing.assert_allclose(residual.var(axis=0), variance, rtol=0.12)


def test_factors_require_jtx_output_mode():
    with pytest.raises(TypeError, match='base_proposal_means'):
        RefinementUpdate(increments=np.zeros((1, 2)), broyden_factors=((),))



def test_nonfinite_transport_falls_back_to_jtx_without_changing_noise():
    def overflowing_transport(req):
        base = picard_jtx_update_fn(req)
        large = np.full_like(req.parent_states[0], 1e308)
        return replace(base, broyden_factors=tuple(
            ((large, large),) for _ in req.nodes
        ))

    paths = []
    for callback in [picard_jtx_update_fn, overflowing_transport]:
        sampler = SpeculativeSampler(
            target=ToyTarget(), proposal=IdentityProposal(),
            tree=DraftTree.chain(3), schedule=ConstantSchedule(0.2), num_steps=3,
            verifier=AcceptFirst(), proposal_refinement_iters=1,
            refinement_update_fn=callback,
        )
        with np.errstate(over='ignore', invalid='ignore'):
            paths.append(sampler.sample(np.ones(2), rng=np.random.default_rng(3)).trajectory)
    np.testing.assert_array_equal(paths[0], paths[1])


@pytest.mark.parametrize('bad', ['rows', 'shape', 'nonfinite'])
def test_malformed_factors_are_rejected(bad):
    def update(req):
        base = picard_jtx_update_fn(req)
        factor = np.zeros_like(req.parent_states[0])
        if bad == 'shape':
            factor = factor[:1]
        if bad == 'nonfinite':
            factor[0] = np.nan
        factors = tuple(((factor, factor),) for _ in req.nodes)
        if bad == 'rows':
            factors = factors[:1]
        return replace(base, broyden_factors=factors)

    sampler = SpeculativeSampler(
        target=ToyTarget(), proposal=IdentityProposal(),
        tree=DraftTree.chain(3), schedule=ConstantSchedule(0.2), num_steps=3,
        verifier=AcceptFirst(), proposal_refinement_iters=1, refinement_update_fn=update,
    )
    with pytest.raises(ValueError):
        sampler.sample(np.ones(2), rng=np.random.default_rng(3))
