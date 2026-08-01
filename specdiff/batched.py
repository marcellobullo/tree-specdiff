"""Algorithm 3 over a batch of trajectories.

The parallelism inside a round -- one batched target call for a whole draft
tree -- is orthogonal to the parallelism across images, and a real generation
job wants both. Doing them together is not just a reshape, because speculation
breaks the property that makes image batching trivial in a standard sampler:
**trajectories accept different prefixes and immediately fall out of step.**
After one round, trajectory 0 may sit at step 4 and trajectory 1 at step 1.

Three consequences, and they are the whole design:

1.  ``sigma`` becomes per-row. Rows of one verification batch belong to
    different steps, so :class:`~specdiff.types.BatchedVerifyRequest` carries
    ``sigmas``, not ``sigma``.
2.  Live rows shrink as the round descends. A trajectory that rejects at level
    1 takes no part in level 2. Rather than mask, the driver *compacts*: each
    level's request contains only rows still walking down their tree, so a rule
    never sees a dead row.
3.  Cost is no longer the mean. One target call serves every live trajectory,
    so the batch advances at the pace of its slowest member and the wall-clock
    speedup is ``N / iterations``, strictly below the mean of what the
    trajectories would achieve alone. :class:`BatchedSamplingResult` reports
    both, plus occupancy, because the gap is the thing you tune batch size
    against.

Trajectory state is held in flat buffers indexed ``row * tree.size + node``, so
the backend needs no gather beyond the row indexing it already had.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple

from .kernels import NoiseSchedule, ProposalTransition, TargetTransition
from .ops import Backend, resolve_backend
from .trees import ROOT, DraftTree
from .types import (
    BatchedRoundRecord,
    BatchedSamplingResult,
    BatchedVerifyRequest,
    BatchedVerifyResult,
)
from .verify import CheckedVerifier, Verifier

Array = Any


# --------------------------------------------------------------------- proposals
class BatchedProposal(ABC):
    """A proposal that knows which trajectory each row belongs to.

    ``slots`` accompanies every call: ``slots[i]`` is the trajectory index of
    row ``i``. Stateless proposals ignore it; a delayed drift uses it to look
    up that trajectory's own frozen drift.
    """

    @abstractmethod
    def means(self, slots: Sequence[int], states: Array, steps: Sequence[int]) -> Array: ...

    def on_round_start(self, slots: Sequence[int], steps: Sequence[int], roots: Array) -> None: ...

    def on_verified(
        self, slots: Sequence[int], steps: Sequence[int], states: Array, target_means: Array
    ) -> None: ...

    def reset(self, num_slots: int) -> None: ...


class StatelessBatchedProposal(BatchedProposal):
    """Adapter lifting any stateless :class:`ProposalTransition` into a batch.

    A draft network, ``IdentityProposal``, ``MirrorProposal``: none of them
    carry per-trajectory state, so all rows go through in one call and slots
    are ignored. Refuses a stateful proposal rather than corrupting it.
    """

    def __init__(self, inner: ProposalTransition) -> None:
        if getattr(inner, "stateful", False):
            raise TypeError(
                f"{type(inner).__name__} is stateful and cannot be shared across a batch: "
                "its cached state belongs to one trajectory. Use a BatchedProposal "
                "(e.g. BatchedDelayedDriftProposal) or wrap per slot with PerSlotProposal."
            )
        self.inner = inner

    def means(self, slots, states, steps):
        return self.inner.means(states, steps)

    def reset(self, num_slots):
        self.inner.reset()


class BatchedDelayedDriftProposal(BatchedProposal):
    """Eq. (7) with root-drift prefetching, one frozen drift per trajectory.

    The increments live in a ``(num_slots, *state_shape)`` buffer, so drafting
    is a single gather-and-add over the whole batch. The warm-up evaluations
    that trajectories need before they hold any drift are collected into one
    batched target call rather than one per trajectory.
    """

    def __init__(self, target: TargetTransition, *, prefetch: bool = True) -> None:
        self._target = target
        self._prefetch = prefetch
        self._increments: Optional[Array] = None
        self._have: List[bool] = []
        self._num_slots = 0

    def reset(self, num_slots: int) -> None:
        self._increments = None
        self._have = [False] * num_slots
        self._num_slots = num_slots

    def on_round_start(self, slots, steps, roots) -> None:
        ops = resolve_backend(roots)
        if self._increments is None:
            self._increments = ops.zeros_stack(self._num_slots, roots[0])
        missing = [i for i, s in enumerate(slots) if self._prefetch is False or not self._have[s]]
        if not missing:
            return
        rows = ops.take(roots, missing)
        means = self._target(rows, tuple(steps[i] for i in missing))
        ops.put(self._increments, [slots[i] for i in missing], means - rows)
        for i in missing:
            self._have[slots[i]] = True

    def on_verified(self, slots, steps, states, target_means) -> None:
        if not self._prefetch:
            return
        ops = resolve_backend(states)
        ops.put(self._increments, list(slots), target_means - states)
        for s in slots:
            self._have[s] = True

    def means(self, slots, states, steps):
        ops = resolve_backend(states)
        return states + ops.take(self._increments, list(slots))


class PerSlotProposal(BatchedProposal):
    """General fallback: one independent proposal instance per trajectory.

    Correct for any stateful proposal, at the cost of a Python loop over the
    slots present in each call. Prefer a native batched implementation when the
    proposal is hot.
    """

    def __init__(self, factory, num_slots: Optional[int] = None) -> None:
        self._factory = factory
        self._instances: List[ProposalTransition] = []
        if num_slots is not None:
            self.reset(num_slots)

    def reset(self, num_slots: int) -> None:
        self._instances = [self._factory() for _ in range(num_slots)]

    def _grouped(self, slots):
        groups: dict[int, List[int]] = {}
        for row, s in enumerate(slots):
            groups.setdefault(s, []).append(row)
        return groups

    def means(self, slots, states, steps):
        ops = resolve_backend(states)
        out = ops.zeros_stack(len(slots), states[0])
        for slot, rows in self._grouped(slots).items():
            sub = ops.take(states, rows)
            ops.put(out, rows, self._instances[slot].means(sub, tuple(steps[r] for r in rows)))
        return out

    def on_round_start(self, slots, steps, roots) -> None:
        for row, slot in enumerate(slots):
            self._instances[slot].on_round_start(steps[row], roots[row])

    def on_verified(self, slots, steps, states, target_means) -> None:
        for row, slot in enumerate(slots):
            self._instances[slot].on_verified(steps[row], states[row], target_means[row])


# ----------------------------------------------------------------------- sampler
class BatchedSpeculativeSampler:
    """Algorithm 3 run on ``batch_size`` independent trajectories at once.

    Same components as :class:`~specdiff.sampler.SpeculativeSampler`, with two
    added requirements:

    * the draft tree must be **level-uniform** (every node at a given depth has
      the same number of children), so that one level's candidates form a
      rectangular ``(batch, K, *shape)`` array;
    * the proposal must be a :class:`BatchedProposal`. Wrap a stateless one in
      :class:`StatelessBatchedProposal`; a stateful one belongs in
      :class:`PerSlotProposal` or gets a native implementation.

    The verifier needs no change: :meth:`Verifier.verify_batch` falls back to a
    row-wise loop over the rule you already have.

    Trajectories are independent, but they share an RNG stream, so a given
    trajectory is *not* bit-reproducible across different batch sizes. Its law
    is unaffected.
    """

    def __init__(
        self,
        target: TargetTransition,
        proposal: BatchedProposal,
        schedule: NoiseSchedule,
        tree: DraftTree,
        verifier: Verifier,
        *,
        num_steps: int,
        check_contract: bool = False,
        keep_trajectories: bool = False,
        backend: Optional[Backend] = None,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if not isinstance(proposal, BatchedProposal):
            raise TypeError(
                "the batched sampler needs a BatchedProposal; wrap a stateless "
                "ProposalTransition in StatelessBatchedProposal"
            )
        if not tree.is_level_uniform():
            raise ValueError(
                f"{tree} is not level-uniform, so its candidates cannot form a rectangular "
                "batch. Use DraftTree.uniform / from_widths, or the single-trajectory sampler."
            )
        verifier.check_topology(tree)
        self.target = target
        self.proposal = proposal
        self.schedule = schedule
        self.tree = tree
        self.verifier = CheckedVerifier(verifier) if check_contract else verifier
        self.num_steps = int(num_steps)
        self.keep_trajectories = keep_trajectories
        self._backend = backend

    # ------------------------------------------------------------------ public
    def sample(self, init: Array, *, rng: Any = None, record: bool = True) -> BatchedSamplingResult:
        """Run from ``init`` of shape ``(batch, *state_shape)``."""
        ops = self._backend or resolve_backend(init)
        ops.check_state_dtype(init, "init")
        batch = int(init.shape[0])
        N = self.num_steps
        size = self.tree.size

        self.target.reset_stats()
        self.proposal.reset(batch)
        self.verifier.reset()

        current = ops.copy(init)
        steps_done = [0] * batch
        rounds_per_traj = [0] * batch
        trajectories = None
        if self.keep_trajectories:
            trajectories = ops.zeros_stack(batch * (N + 1), init[0])
            ops.put(trajectories, [i * (N + 1) for i in range(batch)], current)

        records: List[BatchedRoundRecord] = []
        drafted_total = 0
        iteration = 0

        while any(n < N for n in steps_done):
            active = [i for i in range(batch) if steps_done[i] < N]
            lookaheads = [min(self.tree.depth, N - steps_done[i]) for i in active]
            roots = ops.take(current, active)

            states = ops.zeros_stack(len(active) * size, init[0])
            proposal_means = ops.zeros_stack(len(active) * size, init[0])
            ops.put(states, [r * size + ROOT for r in range(len(active))], roots)

            self.proposal.on_round_start(active, [steps_done[i] for i in active], roots)

            drafted = self._draft(active, lookaheads, states, proposal_means, steps_done, ops, rng)
            target_means, verified = self._verify(active, lookaheads, states, steps_done, ops)
            committed, accepted_depth, rejected = self._accept(
                active,
                lookaheads,
                states,
                proposal_means,
                target_means,
                steps_done,
                trajectories,
                current,
                ops,
                rng,
            )

            record_ = BatchedRoundRecord(
                iteration=iteration,
                active=tuple(active),
                start_steps=tuple(steps_done[i] for i in active),
                committed=tuple(committed),
                drafted=drafted,
                verified=verified,
                accepted_depth=tuple(accepted_depth),
                rejected=tuple(rejected),
            )
            for pos, i in enumerate(active):
                if committed[pos] < 1:
                    raise RuntimeError("a trajectory committed no state; would not terminate")
                steps_done[i] += committed[pos]
                rounds_per_traj[i] += 1
            drafted_total += drafted
            if record:
                records.append(record_)
            iteration += 1

        grouped = None
        if trajectories is not None:
            grouped = ops.group_rows(trajectories, N + 1)

        return BatchedSamplingResult(
            samples=current,
            trajectories=grouped,
            rounds=tuple(records),
            num_steps=N,
            batch_size=batch,
            target_calls=self.target.num_calls,
            target_states_evaluated=self.target.num_states,
            drafted_states=drafted_total,
            rounds_per_trajectory=tuple(rounds_per_traj),
        )

    # ------------------------------------------------------------ phase 1
    def _draft(self, active, lookaheads, states, proposal_means, steps_done, ops, rng) -> int:
        size = self.tree.size
        drafted = 0
        for level in range(1, self.tree.depth + 1):
            rows = [r for r, la in enumerate(lookaheads) if la >= level]
            if not rows:
                break
            parents = [u for u in self.tree.layer(level - 1) if self.tree.children(u)]
            if not parents:
                break

            parent_ids = [r * size + u for r in rows for u in parents]
            parent_slots = [active[r] for r in rows for _ in parents]
            parent_steps = [steps_done[active[r]] + level - 1 for r in rows for _ in parents]

            means = self.proposal.means(parent_slots, ops.take(states, parent_ids), parent_steps)
            ops.put(proposal_means, parent_ids, means)

            counts = [len(self.tree.children(u)) for _ in rows for u in parents]
            child_ids = [
                r * size + v for r in rows for u in parents for v in self.tree.children(u)
            ]
            # sigma is per parent row, and a child inherits its parent's scale
            sigmas = [self.schedule(s) for s in parent_steps]
            noise = ops.randn_stack(len(child_ids), states[0], rng)
            scaled = ops.scale_rows(noise, [s for s, c in zip(sigmas, counts) for _ in range(c)])
            ops.put(states, child_ids, ops.repeat_rows(means, counts) + scaled)
            drafted += len(child_ids)
        return drafted

    # ------------------------------------------------------------ phase 2
    def _verify(self, active, lookaheads, states, steps_done, ops):
        """The single batched target call of the iteration, over every live
        trajectory's internal nodes."""
        size = self.tree.size
        ids, steps = [], []
        for r, la in enumerate(lookaheads):
            for u in self.tree.internal_nodes:
                if self.tree.depth_of(u) < la:  # u is internal in T|_{L_n}
                    ids.append(r * size + u)
                    steps.append(steps_done[active[r]] + self.tree.depth_of(u))
        means = self.target(ops.take(states, ids), tuple(steps))
        buffer = ops.zeros_stack(len(active) * size, states[0])
        ops.put(buffer, ids, means)
        return buffer, len(ids)

    # ------------------------------------------------------------ phase 3
    def _accept(
        self,
        active,
        lookaheads,
        states,
        proposal_means,
        target_means,
        steps_done,
        trajectories,
        current,
        ops,
        rng,
    ) -> Tuple[List[int], List[int], List[bool]]:
        size = self.tree.size
        N = self.num_steps
        cursor = {r: ROOT for r in range(len(active))}
        alive = list(range(len(active)))
        committed = [0] * len(active)
        accepted_depth = [0] * len(active)
        rejected = [False] * len(active)

        for level in range(1, self.tree.depth + 1):
            rows = [r for r in alive if lookaheads[r] >= level]
            if not rows:
                break
            width = self.tree.width_at(level - 1)
            parent_ids = [r * size + cursor[r] for r in rows]
            child_ids = [
                r * size + v for r in rows for v in self.tree.children(cursor[r])
            ]
            steps = [steps_done[active[r]] + level - 1 for r in rows]

            request = BatchedVerifyRequest(
                steps=tuple(steps),
                slots=tuple(active[r] for r in rows),
                proposal_mean=ops.take(proposal_means, parent_ids),
                target_mean=ops.take(target_means, parent_ids),
                sigmas=tuple(self.schedule(s) for s in steps),
                children=ops.group_rows(ops.take(states, child_ids), width),
                parent_state=ops.take(states, parent_ids),
                rng=rng,
                # rows sit at different tree nodes, so the node is a tuple here;
                # BatchedVerifyRequest.row() turns it back into info["node"].
                info={"level": level, "nodes": tuple(cursor[r] for r in rows)},
            )
            result: BatchedVerifyResult = self.verifier.verify_batch(request)

            # commit one state per live row
            ops.put(current, [active[r] for r in rows], result.states)
            if trajectories is not None:
                ops.put(
                    trajectories,
                    [active[r] * (N + 1) + steps_done[active[r]] + level for r in rows],
                    result.states,
                )
            self.proposal.on_verified(
                request.slots, steps, request.parent_state, request.target_mean
            )

            for j, r in enumerate(rows):
                committed[r] += 1
                if result.accepted[j]:
                    accepted_depth[r] += 1
                    cursor[r] = self.tree.children(cursor[r])[result.child_index[j]]
                else:
                    rejected[r] = True
            # Drop the rows that just rejected, in one pass. Removing them
            # individually inside the loop above is a list scan per rejection,
            # i.e. quadratic in batch size on the round where most rows reject
            # -- which is the common one. Order is preserved, and `rows` must
            # keep its ascending order because it indexes the request's rows.
            alive = [r for r in alive if not rejected[r]]
        return committed, accepted_depth, rejected
