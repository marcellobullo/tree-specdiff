"""Run Algorithm 3 over a batch of trajectories.

Within-round parallelism and image batching are independent. Combining them
requires explicit scheduling because trajectories can accept prefixes of
different lengths and advance to different steps.

This behavior has three consequences:

1.  ``sigma`` becomes per-row. Rows of one verification batch belong to
    different steps, so :class:`~specdiff.types.BatchedVerifyRequest` carries
    ``sigmas``, not ``sigma``.
2.  Live rows shrink as the round descends. A trajectory that rejects at level
    1 takes no part in level 2. The sampler compacts active rows at each level,
    so verifiers do not require validity masks.
3.  Cost is no longer the mean. One target call serves every live trajectory,
    so the batch advances at the pace of its slowest member and the wall-clock
    speedup is ``N / iterations``, strictly below the mean of what the
    trajectories would achieve alone. :class:`BatchedSamplingResult` reports
    both values and occupancy to support batch-size selection.

Trajectory state is held in flat buffers indexed ``row * tree.size + node``, so
the backend needs no gather beyond the row indexing it already had.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from .kernels import NoiseSchedule, ProposalTransition, TargetTransition
from .ops import Backend, resolve_backend
from .sampler import PREFETCH_MODES
from .trees import ROOT, DraftTree
from .types import (
    BatchedRoundRecord,
    BatchedSamplingResult,
    BatchedVerifyRequest,
    BatchedVerifyResult,
)
from .verify import CheckedVerifier, Verifier

Array = Any


# ----------------------------------------------------------------------- sampler
class BatchedSpeculativeSampler:
    """Algorithm 3 run on ``batch_size`` independent trajectories at once.

    Uses the same components as :class:`~specdiff.sampler.SpeculativeSampler`,
    with two additional requirements:

    * the draft tree must be **level-uniform** (every node at a given depth has
      the same number of children), so that one level's candidates form a
      rectangular ``(batch, K, *shape)`` array;
    * the proposal must be level-uniform-safe in the same sense; any
      :class:`~specdiff.kernels.ProposalTransition` works unchanged, since the
      interface is the same one the single-image sampler uses.

    Verifiers need no changes because :meth:`Verifier.verify_batch` defaults to
    a row-wise loop over :meth:`Verifier.verify`.

    Trajectories are independent, but they share an RNG stream, so a given
    trajectory is *not* bit-reproducible across different batch sizes. Its law
    is unaffected.

    ``prefetch`` and ``evaluate_leaves`` mean exactly what they mean on
    :class:`~specdiff.sampler.SpeculativeSampler` -- the two samplers resolve
    the same three cases against the same tree -- but here every case is
    decided per row, because rows of one batch sit at different nodes and
    reject at different depths.
    """

    def __init__(
        self,
        target: TargetTransition,
        proposal: ProposalTransition,
        schedule: NoiseSchedule,
        tree: DraftTree,
        verifier: Verifier,
        *,
        num_steps: int,
        check_contract: bool = False,
        keep_trajectories: bool = False,
        prefetch: str = "nearest",
        evaluate_leaves: bool = False,
        backend: Optional[Backend] = None,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if prefetch not in PREFETCH_MODES:
            raise ValueError(
                f"prefetch must be one of {sorted(PREFETCH_MODES)}; got {prefetch!r}. "
                'Use "nearest" for the freshest drift at the committed step, '
                '"parent" for the verified parent\'s, or "none" to re-evaluate '
                "the root."
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
        self.prefetch = prefetch
        self.evaluate_leaves = bool(evaluate_leaves)
        self._check_root_mean = bool(check_contract)
        # batch index -> that row's root target mean, carried from the previous
        # round's committed leaf. Only ever populated under nearest + leaves.
        self._exact_root_means: dict = {}
        self._evaluated_cache: dict = {}
        self._backend = backend

    # ------------------------------------------------------------------ public
    def sample(
        self,
        init: Array,
        *,
        rng: Any = None,
        record: bool = True,
        on_round: Optional[Callable[[int, int], None]] = None,
    ) -> BatchedSamplingResult:
        """Run from ``init`` of shape ``(batch, *state_shape)``.

        ``on_round`` is called after every round with ``(steps_taken, steps_total)``
        -- committed steps summed over the batch, against ``batch * num_steps``.
        It is a progress hook only: rounds commit different numbers of steps per
        trajectory, so the ratio advances unevenly but monotonically.
        """
        ops = self._backend or resolve_backend(init)
        ops.check_state_dtype(init, "init")
        batch = int(init.shape[0])
        N = self.num_steps
        size = self.tree.size

        self.target.reset_stats()
        self.proposal.configure_backend(ops)
        self.proposal.reset(batch)
        self.proposal.configure_prefetch(self.prefetch)
        self.verifier.reset()
        self._exact_root_means = {}

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
            target_means, verified, has_mean = self._verify(
                active, lookaheads, states, steps_done, ops
            )
            committed, accepted_depth, rejected = self._accept(
                active,
                lookaheads,
                states,
                proposal_means,
                target_means,
                has_mean,
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
            if on_round is not None:
                on_round(sum(steps_done), batch * N)

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
            parent_indices = [active[r] for r in rows for _ in parents]
            parent_steps = [steps_done[active[r]] + level - 1 for r in rows for _ in parents]

            means = self.proposal.means(parent_indices, ops.take(states, parent_ids), parent_steps)
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
    def _evaluated_nodes(self, lookahead: int):
        """Nodes of ``T|_{L_n}`` whose target mean this round computes.

        Without ``evaluate_leaves`` that is the internal nodes alone: leaves are
        never parents, so nothing needs their mean to be *verified* (eq. 26,
        ``|I| = B / K``). With it, the leaf level joins the batch -- not to
        verify anything, but so ``prefetch="nearest"`` has an exact drift to
        carry when a round accepts every level. Costs ``K^L`` extra rows; see
        :meth:`DraftTree.verification_budget`.

        Cached on the lookahead, because a round asks once per live row and the
        answer depends on nothing else.
        """
        cached = self._evaluated_cache.get(lookahead)
        if cached is None:
            if self.evaluate_leaves:
                nodes = tuple(u for u in range(self.tree.size)
                              if self.tree.depth_of(u) <= lookahead)
            else:
                nodes = tuple(u for u in self.tree.internal_nodes
                              if self.tree.depth_of(u) < lookahead)
            cached = (nodes, frozenset(nodes))
            self._evaluated_cache[lookahead] = cached
        return cached

    def _verify(self, active, lookaheads, states, steps_done, ops):
        """The single batched target call of the iteration.

        A row whose previous round accepted every level already knows its own
        root's target mean -- the committed leaf's, evaluated last round at this
        very state and this very step -- so that row's root is dropped from the
        batch instead of recomputed. Rows are independent here: one row can skip
        its root while its neighbours pay for theirs.

        Returns the mean buffer, the number of rows actually evaluated, and the
        per-row set of nodes that have a mean, which Phase 3 needs to know which
        drifts it is allowed to carry.
        """
        size = self.tree.size
        ids, steps, indices_in_batch = [], [], []
        has_mean, known_roots = [], []
        for r, la in enumerate(lookaheads):
            known = self._exact_root_means.pop(active[r], None)
            known_roots.append(known)
            nodes, node_set = self._evaluated_nodes(la)
            has_mean.append(node_set)
            for u in nodes:
                if u == ROOT and known is not None:
                    continue
                ids.append(r * size + u)
                steps.append(steps_done[active[r]] + self.tree.depth_of(u))
                # active[r], not r: r is the position in the live list, which
                # shifts as images finish and leave the batch.
                indices_in_batch.append(active[r])

        buffer = ops.zeros_stack(len(active) * size, states[0])
        if ids:
            means = self.target(tuple(indices_in_batch), ops.take(states, ids), tuple(steps))
            ops.put(buffer, ids, means)
        reused = [r for r, known in enumerate(known_roots) if known is not None]
        if reused:
            ops.put(buffer, [r * size + ROOT for r in reused],
                    ops.stack_rows([known_roots[r] for r in reused]))
            if self._check_root_mean:
                self._verify_exact_roots(active, reused, states, buffer, steps_done, ops)
        return buffer, len(ids), has_mean

    def _verify_exact_roots(self, active, reused, states, target_means, steps_done, ops):
        """Validate every reused root mean against a fresh evaluation.

        The batched mirror of :meth:`SpeculativeSampler._verify_exact_root`. The
        reuse rests on an invariant -- that a row's committed leaf last round is
        its root this round, at the same step -- which truncation, topology
        changes, or an indexing slip can violate. Calls ``target.means``
        directly, so the check does not inflate the NFE counters.
        """
        size = self.tree.size
        ids = [r * size + ROOT for r in reused]
        indices = [active[r] for r in reused]
        steps = [steps_done[active[r]] for r in reused]
        fresh = self.target.means(tuple(indices), ops.take(states, ids), tuple(steps))
        for j, r in enumerate(reused):
            if not ops.allclose(fresh[j], target_means[r * size + ROOT]):
                raise ValueError(
                    "the reused root target mean does not match a fresh evaluation "
                    f"for image {active[r]} at step {steps[j]}. The exact-root "
                    "optimisation assumed the previous round's committed leaf is "
                    "this round's root; it is not."
                )

    # ------------------------------------------------------------ phase 3
    def _accept(
        self,
        active,
        lookaheads,
        states,
        proposal_means,
        target_means,
        has_mean,
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
        # Under "nearest" the hand-over is deferred to the end of the round, so
        # each row remembers where it last committed and from which parent.
        last_parent = [ROOT] * len(active)
        last_state = [None] * len(active)

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
                indices_in_batch=tuple(active[r] for r in rows),
                proposal_mean=ops.take(proposal_means, parent_ids),
                target_mean=ops.take(target_means, parent_ids),
                sigmas=tuple(self.schedule(s) for s in steps),
                children=ops.group_rows(ops.take(states, child_ids), width),
                parent_state=ops.take(states, parent_ids),
                rng=rng,
                backend=ops,
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
            if self.prefetch == "parent":
                # This parent's target mean was paid for in Phase 2 and is the
                # freshest drift in existence -- but it was computed at step
                # n + level - 1 from the parent, while the next round starts at
                # the committed child. Handed over whether or not the child was
                # accepted: the evaluation was made either way, and on a level-1
                # rejection it is the only drift the next round will have.
                self.proposal.on_verified(
                    request.indices_in_batch, steps, request.parent_state,
                    request.target_mean,
                )
            elif self.prefetch == "nearest":
                # Defer: a drift evaluated *at* the committed step only becomes
                # selectable once the round has stopped descending. See
                # `_carry_nearest`.
                for j, r in enumerate(rows):
                    last_parent[r] = cursor[r]
                    last_state[r] = result.states[j]

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

        if self.prefetch == "nearest":
            self._carry_nearest(active, states, target_means, has_mean, cursor,
                                last_parent, last_state, committed, rejected,
                                steps_done, ops)
        return committed, accepted_depth, rejected

    def _carry_nearest(self, active, states, target_means, has_mean, terminal,
                       last_parent, last_state, committed, rejected, steps_done, ops):
        """Hand every live row the freshest drift available at its committed step.

        The batched mirror of :meth:`SpeculativeSampler._prefetch_nearest`: the
        same three cases, resolved independently per row because rows end the
        round at different nodes and different depths. All candidate drifts were
        evaluated in Phase 2, so this adds no target evaluations -- and the rows
        are handed over in one call, so the proposal sees one update per round
        rather than one per row.
        """
        size = self.tree.size
        indices, steps, chosen = [], [], []
        for r in range(len(active)):
            if not committed[r]:
                continue
            n = steps_done[active[r]]
            parent = last_parent[r]
            depth = self.tree.depth_of(parent) + 1
            node = terminal[r]

            if not rejected[r] and node in has_mean[r]:
                # Case 1: full acceptance, and the committed leaf's own drift was
                # computed in Phase 2. It is exact at the next root -- same state,
                # same step -- so record it and let that row drop its root from
                # the next batch.
                pick, step = node, n + self.tree.depth_of(node)
                self._exact_root_means[active[r]] = ops.copy(target_means[r * size + pick])
            else:
                usable = [v for v in self.tree.children(parent) if v in has_mean[r]]
                if rejected[r] and usable:
                    # Case 2: the committed state is a residual draw, not a
                    # drafted node, so no exact drift exists. Its siblings do sit
                    # at the same step and were verified in Phase 2; carry the
                    # nearest one's. The residual is drawn close to the drafts by
                    # construction, which is what makes this a useful proxy.
                    pick, best = usable[0], None
                    for v in usable:
                        d = ops.norm(states[r * size + v] - last_state[r])
                        if best is None or d < best:
                            pick, best = v, d
                    step = n + depth
                else:
                    # Case 3: rejection at the last level (siblings are leaves
                    # and were not evaluated), or anything else unavailable --
                    # the parent's, one step stale.
                    pick, step = parent, n + depth - 1
            indices.append(active[r])
            steps.append(step)
            chosen.append(r * size + pick)

        if indices:
            self.proposal.on_verified(
                tuple(indices), tuple(steps), ops.take(states, chosen),
                ops.take(target_means, chosen),
            )
