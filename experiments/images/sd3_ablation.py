"""Controlled SD3 carry experiments; used by notebooks/sd3-carry-ablation.ipynb.

This is an experimental adapter, not a change to the production sampler. Keyed
streams isolate images, round starts, draft nodes, and verification levels.
Never key a draft solely by its destination timestep: discarded speculative
noise must not be reused in a later round after conditioning on rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Sequence

import torch

from specdiff import BatchedSpeculativeSampler, DelayedDriftProposal, DraftTree, create_verifier
from specdiff.refinement import ExactTargetMean
from specdiff.verify import Verifier
from . import sd3_models as sd3


CARRY_POLICIES = ("parent", "nearest", "nearest-parent", "nearest-all")
RNG_MODES = ("stream", "keyed")


class RandomTape:
    """Stable per-event seeds, shared across treatments but never reused in a run.

    'verify' seeds a local sequence of uniforms for one visited level. Different
    rules need not assign the same meaning to its draw positions. Compare carry
    policies within a rule first. Reproducibility requires the same RNG backend,
    device and dtype; it does not imply bitwise-identical GPU model arithmetic.
    """

    def __init__(self, seed: int, *, record: bool = False):
        self.seed = int(seed)
        self.record = record
        self.events = []
        self._seen = set()

    def seed_for(self, image: int, start: int, phase: str, slot=()) -> int:
        key = ["sd3-ablation-v1", self.seed, int(image), int(start), phase, slot]
        digest = hashlib.blake2b(json.dumps(key, separators=(",", ":")).encode(), digest_size=8)
        return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)

    def generator(self, image: int, start: int, phase: str, slot=(), *, device):
        key = (int(image), int(start), phase, json.dumps(slot))
        if key in self._seen:
            raise RuntimeError(f"Random event reused in the same run: {key}")
        self._seen.add(key)
        seed = self.seed_for(image, start, phase, slot)
        if self.record:
            self.events.append(dict(image_id=int(image), start_step=int(start),
                                    phase=phase, slot=json.dumps(slot), event_seed=seed))
        return torch.Generator(device=device).manual_seed(seed)


class _DraftBackend:
    """Replace only draft innovations; leave the production drafting logic intact."""

    def __init__(self, base, sampler, active, lookaheads, steps_done):
        self.base, self.sampler = base, sampler
        self.active, self.lookaheads, self.steps_done = active, lookaheads, steps_done
        self.level = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def randn_stack(self, n, ref, rng=None):
        self.level += 1
        s = self.sampler
        children = [v for u in s.tree.layer(self.level - 1) for v in s.tree.children(u)]
        draws = []
        for r, image in enumerate(self.active):
            if self.lookaheads[r] < self.level:
                continue
            for node in children:
                gen = s.tape.generator(s.image_ids[image], self.steps_done[image],
                                       "draft", s.paths[node], device=ref.device)
                draws.append(self.base.randn_stack(1, ref, gen)[0])
        if len(draws) != n:
            raise RuntimeError("Production draft layout changed; update the experimental adapter")
        return self.base.stack_rows(draws)


class _EventVerifier(Verifier):
    def __init__(self, inner, sampler):
        self.inner, self.sampler = inner, sampler
        self.max_children = inner.max_children
        self.name = inner.name

    def reset(self):
        self.inner.reset()

    def verify(self, request):
        s = self.sampler
        image = request.index_in_batch
        start = s.round_starts[image]
        level = request.step - start
        if s.rng_mode == "keyed":
            gen = s.tape.generator(s.image_ids[image], start, "verify", level,
                                   device=request.children.device)
            request = replace(request, rng=gen)
        result = self.inner.verify(request)
        ops = self.backend_for(request)
        s.verification_log.append(dict(
            image_id=s.image_ids[image], start_step=start, step=request.step,
            level=level + 1, node=request.info["node"],
            delta=ops.norm(request.target_mean - request.proposal_mean) / request.sigma,
            accepted=bool(result.accepted), child_index=result.child_index,
        ))
        return result


def choose_carry(tree, policy, states, known, parent, terminal, committed, rejected, ops):
    """Select only evaluated nodes. Preserve an exact accepted leaf when available."""
    if policy not in CARRY_POLICIES:
        raise ValueError(f"Unknown carry policy: {policy}")
    if policy == "parent":
        return parent, False
    if not rejected and terminal in known:
        return terminal, True
    siblings = [v for v in tree.children(parent) if v in known]
    if policy == "nearest-all":
        candidates = sorted(known)
    elif policy == "nearest-parent":
        candidates = sorted(set(siblings + [parent]))
    else:
        candidates = siblings if rejected and siblings else [parent]
    # Tie-breaking is deterministic (lowest node id) for the expanded policies.
    pick = min(candidates, key=lambda v: ops.norm(states[v] - committed))
    return pick, False


class ControlledSampler(BatchedSpeculativeSampler):
    """Torch-only diagnostic sampler, with production verification and drafting."""

    def __init__(self, *args, seed, image_ids, rng_mode="keyed", carry="nearest",
                 diagnose_drift=False, record_random=False, **kwargs):
        if rng_mode not in RNG_MODES or carry not in CARRY_POLICIES:
            raise ValueError("Unknown randomness mode or carry policy")
        if kwargs.get("proposal_refinement_iters"):
            raise ValueError("This ablation holds proposal refinement disabled")
        self.master_seed = int(seed)
        self.image_ids = tuple(int(i) for i in image_ids)
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must be unique")
        self.rng_mode, self.carry = rng_mode, carry
        self.diagnose_drift, self.record_random = diagnose_drift, record_random
        super().__init__(*args, prefetch="nearest", **kwargs)
        self.verifier = _EventVerifier(self.verifier, self)

    def sample(self, init, *, rng=None, **kwargs):
        if not isinstance(init, torch.Tensor) or len(init) != len(self.image_ids):
            raise ValueError("Supply a Torch batch with one unique image_id per row")
        if self.rng_mode == "stream" and rng is None:
            raise ValueError("stream mode requires an explicitly seeded generator")
        self.tape = RandomTape(self.master_seed, record=self.record_random)
        self.carry_log, self.verification_log = [], []
        self.diagnostic_target_calls = self.diagnostic_target_rows = 0
        return super().sample(init, rng=rng, **kwargs)

    def _draft(self, active, lookaheads, states, proposal_means, scaled_innovations,
               steps_done, ops, rng):
        self.round_starts = {i: steps_done[i] for i in active}
        if self.rng_mode == "keyed":
            if not hasattr(self, "paths"):
                self.paths = {0: ()}
                for node in range(1, self.tree.size):
                    parent = self.tree.parent(node)
                    self.paths[node] = self.paths[parent] + (self.tree.children(parent).index(node),)
            ops = _DraftBackend(ops, self, active, lookaheads, steps_done)
        return super()._draft(active, lookaheads, states, proposal_means,
                              scaled_innovations, steps_done, ops, rng)

    def _carry_nearest(self, active, states, target_means, has_mean, terminal,
                       last_parent, last_state, committed, rejected, steps_done, ops):
        size = self.tree.size
        indices, steps, chosen, diagnostic = [], [], [], []
        for r, image in enumerate(active):
            end = steps_done[image] + committed[r]
            if not committed[r] or end >= self.num_steps:
                continue
            start, parent = steps_done[image], last_parent[r]
            row = states[r * size:(r + 1) * size]
            pick, exact = choose_carry(self.tree, self.carry, row, has_mean[r], parent,
                                      terminal[r], last_state[r], rejected[r], ops)
            step = start + self.tree.depth_of(pick)
            if exact:
                self._exact_root_means[image] = ExactTargetMean(
                    target=self.target, index_in_batch=image, step=step,
                    state=ops.copy(row[pick]), mean=ops.copy(target_means[r * size + pick]))
            indices.append(image)
            steps.append(step)
            chosen.append(r * size + pick)
            depth = self.tree.depth_of(parent) + 1
            reason = ("full_accept" if not rejected[r] else
                      "terminal_reject" if depth == min(self.tree.depth, self.num_steps - start)
                      else "internal_reject")
            entry = dict(image_id=self.image_ids[image], start_step=start, end_step=end,
                         reason=reason, parent_node=parent, selected_node=pick,
                         selected_depth=self.tree.depth_of(pick),
                         timestep_offset=step - end, selected_parent=pick == parent,
                         exact=exact, distance=ops.norm(row[pick] - last_state[r]),
                         parent_distance=ops.norm(row[parent] - last_state[r]))
            self.carry_log.append(entry)
            diagnostic.append((r, image, end, entry))
        if self.diagnose_drift and diagnostic:
            self._diagnose_carry(diagnostic, states, target_means, has_mean, terminal,
                                 last_parent, last_state, rejected, steps_done, ops)
        if indices:
            self.proposal.on_verified(tuple(indices), tuple(steps), ops.take(states, chosen),
                                      ops.take(target_means, chosen))

    def _diagnose_carry(self, rows, states, means, known, terminal, parents,
                        committed, rejected, starts, ops):
        # Direct means(), rather than __call__(), deliberately leaves sampler
        # NFE/cache unchanged. Record this additional work separately and do
        # not compare wall times to runs with diagnostics disabled.
        images = tuple(image for _, image, _, _ in rows)
        steps = tuple(end for _, _, end, _ in rows)
        current = ops.stack_rows([committed[r] for r, _, _, _ in rows])
        truth = self.target.means(images, current, steps)
        self.diagnostic_target_calls += 1
        self.diagnostic_target_rows += len(rows)
        for j, (r, image, end, entry) in enumerate(rows):
            offset = r * self.tree.size
            tree_states = states[offset:offset + self.tree.size]
            for policy in CARRY_POLICIES:
                pick, _ = choose_carry(self.tree, policy, tree_states, known[r], parents[r],
                                       terminal[r], committed[r], rejected[r], ops)
                old_step = starts[image] + self.tree.depth_of(pick)
                drift = self.target.freeze_drift(states[offset+pick:offset+pick+1],
                                                 means[offset+pick:offset+pick+1], (old_step,))
                proposal = self.target.apply_drift(drift, current[j:j+1], (end,))[0]
                error = ops.norm(proposal - truth[j])
                entry[f"error_{policy}"] = error
                sigma = self.schedule(end)
                entry[f"delta_{policy}"] = error / sigma if sigma > 0 else None


@dataclass
class ArmResult:
    images: list = field(default_factory=list)
    carries: list = field(default_factory=list)
    verifications: list = field(default_factory=list)
    random_events: list = field(default_factory=list)
    latents: object = None
    diagnostic_target_calls: int = 0
    diagnostic_target_rows: int = 0


@torch.no_grad()
def run_arm(denoiser, *, image_ids: Sequence[int], seed: int, rule="paws",
            match="verification", carry="nearest", rng_mode="keyed", sample_batch=1,
            forward_batch=16, num_steps=50, eps=.8, shift=3., s_noise=1.,
            K=2, L=3, evaluate_leaves=None, diagnose_drift=False, record_random=False):
    """Run one treatment with fixed prompts, schedule and model already loaded.

    Here the experiment explicitly maps verification -> no leaves and budget
    -> leaves. In the library itself `match` only sizes chain verifiers. Passing
    evaluate_leaves explicitly is useful for the match-label negative control.
    No output reuse, downloads, image decoding, or file writes happen here.
    """
    image_ids = tuple(int(i) for i in image_ids)
    if not image_ids or len(set(image_ids)) != len(image_ids) or sample_batch < 1:
        raise ValueError("Supply nonempty unique image_ids and sample_batch >= 1")
    if any(i < 0 or i >= len(denoiser.prompts) for i in image_ids):
        raise ValueError("image_ids must index the denoiser's prompt table")
    if match not in ("verification", "budget"):
        raise ValueError("Unknown matching mode")
    leaves = match == "budget" if evaluate_leaves is None else bool(evaluate_leaves)
    setting = sd3.build(denoiser, num_steps=num_steps, eps=eps, shift=shift,
                        s_noise=s_noise, forward_batch=forward_batch)
    tree = create_verifier(rule).matched_tree(DraftTree.uniform(K, L),
            num_steps=num_steps, match=match, evaluate_leaves=leaves)
    out = ArmResult()
    samples = []
    common = dict(seed=int(seed), rule=rule, match=match, carry=carry, rng_mode=rng_mode,
                  evaluate_leaves=leaves, sample_batch=sample_batch, forward_batch=forward_batch,
                  K=K, L=L, num_steps=num_steps, eps=eps, shift=shift, s_noise=s_noise,
                  tree_depth=tree.depth, proposal_budget=tree.budget,
                  target_node_budget=tree.verification_budget(evaluate_leaves=leaves))
    for first in range(0, len(image_ids), sample_batch):
        ids = image_ids[first:first + sample_batch]
        denoiser.set_prompt_batch(ids)
        tape = RandomTape(seed, record=record_random)
        # Initial states are identical across both RNG modes and every treatment.
        init = torch.stack([torch.randn(setting.state_shape, dtype=torch.float32,
            device=denoiser.device, generator=tape.generator(i, -1, "initial", device=denoiser.device))
            for i in ids])
        stream = (tape.generator(ids[0], -1, "stream", device=denoiser.device)
                  if rng_mode == "stream" else None)
        sampler = ControlledSampler(target=setting.target,
            proposal=DelayedDriftProposal(setting.target), schedule=setting.schedule,
            tree=tree, verifier=create_verifier(rule), num_steps=num_steps,
            evaluate_leaves=leaves, seed=seed, image_ids=ids, rng_mode=rng_mode,
            carry=carry, diagnose_drift=diagnose_drift, record_random=record_random)
        result = sampler.sample(init, rng=stream)
        samples.append(result.samples.cpu())
        for local, image in enumerate(ids):
            records = [(rec, rec.active.index(local)) for rec in result.rounds if local in rec.active]
            accepted = sum(rec.accepted_depth[j] for rec, j in records)
            calls = result.target_calls_per_trajectory[local]
            out.images.append(dict(common, image_id=image,
                target_calls=calls, target_states=result.target_states_per_trajectory[local],
                rounds=result.rounds_per_trajectory[local], speedup=num_steps / max(calls, 1),
                acceptance_rate=accepted / num_steps))
        for target, rows in ((out.carries, sampler.carry_log),
                             (out.verifications, sampler.verification_log),
                             (out.random_events, tape.events + sampler.tape.events)):
            target.extend(dict(common, **row) for row in rows)
        out.diagnostic_target_calls += sampler.diagnostic_target_calls
        out.diagnostic_target_rows += sampler.diagnostic_target_rows
    out.latents = torch.cat(samples)
    return out
