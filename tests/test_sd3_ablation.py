"""Invariants needed before interpreting the carry ablation scientifically."""

import pytest
import torch

from specdiff import (BatchedSpeculativeSampler, ConstantSchedule, DelayedDriftProposal,
                      DraftTree, TargetTransition, create_verifier)
from specdiff.ops import TorchBackend
from experiments.images.sd3_ablation import (ControlledSampler, RandomTape, choose_carry,
                                              run_arm)
from experiments.images.sd3_models import SD3Denoiser
from experiments.images.toy_sd3 import ToySD3Pipeline


class LinearTarget(TargetTransition):
    def means(self, indices_in_batch, states, steps):
        return .7 * states + .2

    def freeze_drift(self, states, means, steps):
        return means - states

    def apply_drift(self, drift, states, steps):
        return states + drift


def sampler(*, controlled=True, rule="d-grs", leaves=True, carry="nearest", **kw):
    target = LinearTarget()
    args = dict(target=target, proposal=DelayedDriftProposal(target),
                tree=DraftTree.uniform(2, 3), verifier=create_verifier(rule),
                schedule=ConstantSchedule(.4), num_steps=9, evaluate_leaves=leaves,
                keep_trajectories=True)
    if controlled:
        return ControlledSampler(**args, carry=carry, **kw)
    return BatchedSpeculativeSampler(**args, prefetch=carry)


@pytest.mark.parametrize("rule", ["paws", "d-grs"])
@pytest.mark.parametrize("leaves", [False, True])
@pytest.mark.parametrize("carry", ["parent", "nearest"])
def test_stream_adapter_preserves_production_sampler(rule, leaves, carry):
    init = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10
    plain = sampler(controlled=False, rule=rule, leaves=leaves, carry=carry)
    wrapped = sampler(rule=rule, leaves=leaves, carry=carry, seed=17,
                      image_ids=[0, 1, 2], rng_mode="stream")
    a = plain.sample(init, rng=torch.Generator().manual_seed(90))
    b = wrapped.sample(init, rng=torch.Generator().manual_seed(90))
    assert torch.equal(a.trajectories, b.trajectories)
    assert a.target_calls_per_trajectory == b.target_calls_per_trajectory
    assert a.target_states_per_trajectory == b.target_states_per_trajectory


@pytest.mark.parametrize("carry", ["parent", "nearest", "nearest-parent", "nearest-all"])
def test_keyed_streams_survive_rebatching_and_permutation(carry):
    ids = [40, 2, 9]
    init = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10
    whole = sampler(seed=17, image_ids=ids, carry=carry)
    a = whole.sample(init)
    for j in [2, 0, 1]:
        single = sampler(seed=17, image_ids=[ids[j]], carry=carry)
        b = single.sample(init[j:j+1])
        assert torch.equal(a.trajectories[j], b.trajectories[0])
        assert a.target_calls_per_trajectory[j] == b.target_calls_per_trajectory[0]
    assert any(len(set(rec.start_steps)) > 1 for rec in a.rounds)


def test_unused_verifier_draws_cannot_shift_another_event():
    a, b = RandomTape(1), RandomTape(1)
    torch.rand(100, generator=a.generator(7, 0, "verify", 0, device="cpu"))
    torch.rand(1, generator=b.generator(7, 0, "verify", 0, device="cpu"))
    for phase, slot in [("draft", (0, 1)), ("verify", 1)]:
        x = torch.randn(8, generator=a.generator(7, 3, phase, slot, device="cpu"))
        y = torch.randn(8, generator=b.generator(7, 3, phase, slot, device="cpu"))
        assert torch.equal(x, y)
    assert a.seed_for(7, 0, "draft", (0, 0)) != a.seed_for(7, 1, "draft", (0,))
    with pytest.raises(RuntimeError, match="reused"):
        a.generator(7, 3, "verify", 1, device="cpu")


def test_nearest_all_uses_other_branches_but_never_unevaluated_nodes():
    tree, ops = DraftTree.uniform(2, 3), TorchBackend()
    states = torch.arange(15, dtype=torch.float64).reshape(-1, 1) + 10
    states[2] = .1
    states[3] = .3
    states[7], states[8] = .2, .4
    states[14] = 0  # closest but NOT evaluated
    committed, known = torch.zeros(1, dtype=torch.float64), {0, 1, 2, 3, 7, 8}
    select = lambda policy: choose_carry(tree, policy, states, known, 3, 3, committed, True, ops)
    assert select("nearest") == (7, False)
    assert select("nearest-all") == (2, False)
    states[3] = .05
    assert select("nearest-parent") == (3, False)
    assert select("nearest-all") == (3, False)
    states[2] = states[7]
    assert choose_carry(tree, "nearest-all", states, known, 3, 7, states[7], False, ops) == (7, True)


def test_oracle_diagnostics_do_not_change_trajectories_or_sampling_cost():
    init = torch.zeros((3, 4), dtype=torch.float64)
    a = sampler(seed=5, image_ids=[3, 6, 8], carry="nearest-all")
    b = sampler(seed=5, image_ids=[3, 6, 8], carry="nearest-all", diagnose_drift=True)
    ra, rb = a.sample(init), b.sample(init)
    assert torch.equal(ra.trajectories, rb.trajectories)
    assert ra.target_calls == rb.target_calls
    assert ra.target_states_evaluated == rb.target_states_evaluated
    assert b.diagnostic_target_rows == len(b.carry_log) > 0
    assert all("error_nearest-all" in entry for entry in b.carry_log)


def test_sd3_match_label_control_and_direct_nfe_accounting():
    den = SD3Denoiser(ToySD3Pipeline(resolution_px=32), ["cat", "dog", "boat"],
                      guidance_scale=1., resolution_px=32)
    args = dict(image_ids=[2, 0], seed=3, num_steps=8, eps=.8,
                rule="paws", evaluate_leaves=False, record_random=True)
    a = run_arm(den, match="verification", sample_batch=2, **args)
    b = run_arm(den, match="budget", sample_batch=1, **args)
    assert torch.equal(a.latents, b.latents)
    assert [r['target_calls'] for r in a.images] == [r['target_calls'] for r in b.images]
    assert all(row['speedup'] == 8 / row['target_calls'] for row in a.images)
    events = lambda run: {(r['image_id'], r['start_step'], r['phase'], r['slot']): r['event_seed']
                          for r in run.random_events}
    assert events(a) == events(b)
    assert all(r['target_node_budget'] == 7 for r in a.images)
    c = run_arm(den, image_ids=[0], seed=3, num_steps=8, match="budget")
    assert c.images[0]['target_node_budget'] == 15


def test_keyed_nearest_all_preserves_a_known_gaussian_target_law():
    # This catches accidental reuse of conditioned-on discarded innovations.
    # Each global image id is an independent random stream.
    count = 1200
    s = sampler(seed=813, image_ids=range(count), carry="nearest-all")
    result = s.sample(torch.zeros((count, 2), dtype=torch.float64))
    expected_mean = .2 * (1 - .7 ** 9) / (1 - .7)
    expected_var = .4 ** 2 * (1 - .7 ** 18) / (1 - .7 ** 2)
    mean_error = torch.abs(result.samples.mean(dim=0) - expected_mean)
    assert torch.all(mean_error < 5 * (expected_var / count) ** .5)
    assert torch.all(torch.abs(result.samples.var(dim=0) - expected_var) < .06)


@pytest.mark.parametrize("controlled", [False, True], ids=["production", "experiment"])
@pytest.mark.parametrize("rejection_depth", [1, 2])
@pytest.mark.parametrize("nearest_child", [0, 1])
def test_internal_rejection_nearest_is_identical_with_and_without_leaves(
        controlled, rejection_depth, nearest_child):
    """Extra leaf means cannot change an internal-level nearest selection.

    Hold drafted states and residuals fixed; use the real evaluation mask and
    carry implementation, including compacted rows at different starting steps.
    Put an evaluated leaf exactly at each residual to ensure nearest does not
    accidentally search beyond the last parent's children.
    """
    class Capture:
        def on_verified(self, indices, steps, states, means):
            self.record = (indices, steps, states.clone(), means.clone())

    active, starts, ops = [2, 0], [3, 0, 1], TorchBackend()
    tree = DraftTree.uniform(2, 3)
    parents = [0, 0] if rejection_depth == 1 else [1, 2]
    states = torch.arange(2 * tree.size * 2, dtype=torch.float64).reshape(-1, 2) / 10
    chosen = [r * tree.size + tree.children(p)[nearest_child] for r, p in enumerate(parents)]
    residuals = [states[node].clone() + .01 for node in chosen]
    for r, residual in enumerate(residuals):
        states[r * tree.size + tree.layer(3)[0]] = residual

    records = []
    for leaves in (False, True):
        options = dict(seed=17, image_ids=[90, 91, 92]) if controlled else {}
        s = sampler(controlled=controlled, leaves=leaves, carry="nearest", **options)
        s.proposal = Capture()
        s.carry_log = []
        means, _, known, _ = s._verify(active, [3, 3], states, starts, ops, None)
        for r, parent in enumerate(parents):
            assert all(v in known[r] for v in tree.children(parent))
        s._carry_nearest(active, states, means, known, dict(enumerate(parents)),
                         parents, residuals, [rejection_depth] * 2, [True] * 2,
                         starts, ops)
        record = s.proposal.record
        assert record[0] == tuple(active)
        assert record[1] == tuple(starts[i] + rejection_depth for i in active)
        assert torch.equal(record[2], states[chosen])
        assert not s._exact_root_means  # Residuals must never be marked exact.
        if controlled:
            assert all(row["reason"] == "internal_reject" for row in s.carry_log)
        records.append(record)
    assert records[0][:2] == records[1][:2]
    assert torch.equal(records[0][2], records[1][2])
    assert torch.equal(records[0][3], records[1][3])
