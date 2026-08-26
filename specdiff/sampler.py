"""Algorithm 3: speculative diffusion sampling for an arbitrary draft tree.

The implementation follows the pseudocode, with line references in comments.
It owns the execution loop and accounting; topology, transition models, and
verification are defined in separate modules.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from .kernels import NoiseSchedule, ProposalTransition, TargetTransition
from .ops import Backend, resolve_backend
from .trees import ROOT, DraftTree
from .types import RoundRecord, SamplingResult, VerifyRequest, VerifyResult
from .verify import CheckedVerifier, Verifier

Array = Any
RoundCallback = Callable[[RoundRecord], None]


PREFETCH_MODES = ("none", "parent", "nearest")
"""Which already-computed drift the next round reuses (Appendix C).

``"none"``
    Reuse nothing: re-evaluate the target at every round's root. Costs one
    extra NFE per round and provides an exact root drift.
``"parent"``
    The last verified parent's drift. Free, but one step behind the state the
    next round starts from: it was computed at step ``n + level - 1`` from the
    parent, while the round begins at ``n + level`` from the child.
``"nearest"``
    The freshest drift available *at the committed step*, still free: the
    committed leaf's own when ``evaluate_leaves`` is on (exact), else the
    nearest drafted sibling at that depth, else the parent.

Every mode preserves exactness. The proposal affects which states are drafted,
while the verifier ensures that committed states follow the target distribution.
The modes trade acceptance rate against target-evaluation cost.
"""


class SpeculativeSampler:
    """Draft-tree speculative sampler.

    Parameters
    ----------
    target, proposal:
        The mean maps ``m^q`` and ``m^p`` of eq. (24).
    schedule:
        The shared scales ``{sigma_n}``.
    tree:
        Draft topology. ``DraftTree.uniform(K, L)`` for the paper's family,
        ``DraftTree.chain(L)`` for a linear lookahead.
    verifier:
        The pluggable rule. Its topology constraints are checked here, at
        construction, not per node.
    num_steps:
        Horizon ``N``.
    check_contract:
        Wrap the verifier in :class:`~specdiff.verify.CheckedVerifier`.
        Recommended while developing a new rule.

    Notes
    -----
    One instance samples one trajectory at a time. The parallelism the method
    exploits is *within* a round: the draft tree passes through the
    target model in a single batched call -- which is a different axis from
    batching over images. Batching trajectories as well is possible but not
    automatic because independent trajectories can accept different prefix
    lengths. See :class:`specdiff.batched.BatchedSpeculativeSampler`.
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
        prefetch: str = "parent",
        evaluate_leaves: bool = False,
        backend: Optional[Backend] = None,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if prefetch not in PREFETCH_MODES:
            raise ValueError(
                f"prefetch must be one of {sorted(PREFETCH_MODES)}; got {prefetch!r}. "
                'Use "parent" to reuse the verified parent drift or "none" '
                "to re-evaluate the root."
            )
        verifier.check_topology(tree)
        self.target = target
        self.proposal = proposal
        self.schedule = schedule
        self.tree = tree
        self.verifier = CheckedVerifier(verifier) if check_contract else verifier
        self.num_steps = int(num_steps)
        self.prefetch = prefetch
        self.evaluate_leaves = bool(evaluate_leaves)
        self._check_root_mean = bool(check_contract)
        self._exact_root_mean = None
        self._backend = backend

    # ------------------------------------------------------------------ public
    def sample(
        self,
        init: Array,
        *,
        rng: Any = None,
        on_round: Optional[RoundCallback] = None,
        record: bool = True,
    ) -> SamplingResult:
        """Run the sampler from ``Y_0 = init`` and return the trajectory.

        ``init`` is a single state of shape ``state_shape``; the caller draws it
        from ``q_0`` (line 1), because the initial distribution is a property of
        the model, not of the sampler.
        """
        ops = self._backend or resolve_backend(init)
        ops.check_state_dtype(init, "init")
        self.target.reset_stats()
        self.proposal.configure_backend(ops)
        self.proposal.reset(1)
        self.proposal.configure_prefetch(self.prefetch)
        self._exact_root_mean = None
        self.verifier.reset()

        N = self.num_steps
        trajectory = ops.zeros_stack(N + 1, init)
        ops.put(trajectory, [0], init[None])

        records: List[RoundRecord] = []
        drafted_total = 0
        n = 0  # line 1
        while n < N:  # line 2
            state = trajectory[n]
            record_ = self._round(n, state, trajectory, ops, rng)
            if record_.committed < 1:  # defensive: a round must always commit
                raise RuntimeError("verification rule committed no state; would not terminate")
            n += record_.committed  # line 24
            drafted_total += record_.drafted
            if record:
                records.append(record_)
            if on_round is not None:
                on_round(record_)

        return SamplingResult(
            trajectory=trajectory,
            rounds=tuple(records),
            num_steps=N,
            target_calls=self.target.num_calls,
            target_states_evaluated=self.target.num_states,
            drafted_states=drafted_total,
        )

    # ----------------------------------------------------------------- one round
    def _round(self, n: int, root_state: Array, trajectory: Array, ops: Backend, rng) -> RoundRecord:
        # line 3: L_n = min(L, N - n); the tree is truncated for the final rounds
        lookahead = min(self.tree.depth, self.num_steps - n)
        tree = self.tree.truncate(lookahead)

        states = ops.zeros_stack(tree.size, root_state)
        ops.put(states, [ROOT], root_state[None])
        proposal_means = ops.zeros_stack(tree.size, root_state)

        # One image is batch_size = 1: every entry belongs to image 0.
        self.proposal.on_round_start((0,), (n,), ops.stack_rows([root_state]))

        # ---------------------------------------------------- Phase 1: drafting
        # lines 5-9. Sequential in depth (a child cannot precede its parent),
        # parallel within a level. Every sibling is expanded, because which
        # child survives is only decided in Phase 3.
        for level in range(1, lookahead + 1):
            parents = [u for u in tree.layer(level - 1) if tree.children(u)]
            if not parents:
                break
            step = n + level - 1
            sigma = self.schedule(step)
            means = self.proposal.means(
                (0,) * len(parents), ops.take(states, parents), (step,) * len(parents)
            )
            ops.put(proposal_means, parents, means)

            counts = [len(tree.children(u)) for u in parents]
            child_ids = [v for u in parents for v in tree.children(u)]
            noise = ops.randn_stack(len(child_ids), root_state, rng)
            ops.put(states, child_ids, ops.repeat_rows(means, counts) + sigma * noise)

        # ------------------------------------------------ Phase 2: verification
        # lines 11-13. The single target evaluation of the round, batched over
        # the internal nodes only: leaves are never parents, so their target
        # means are never needed (eq. 26: |I| = B / K).
        # A round that follows a fully-accepted one under `prefetch="nearest"`
        # with `evaluate_leaves` already knows its own root's target mean: the
        # committed leaf's, computed last round at this very state and step.
        # Re-evaluating it would return the same number, so drop it from the
        # batch -- one row per round, and the only case where skipping is
        # provably exact rather than an approximation.
        known_root_mean, self._exact_root_mean = self._exact_root_mean, None

        internal = tree.internal_nodes
        if self.evaluate_leaves:
            # Leaves are never parents, so nothing needs their target mean to
            # be *verified*. They are evaluated only so `prefetch="nearest"`
            # has an exact drift to carry when the round accepts every level.
            # Costs K^L extra rows; see DraftTree.verification_budget.
            evaluated = internal + tuple(u for u in range(tree.size)
                                         if u not in set(internal))
        else:
            evaluated = internal
        has_mean = set(evaluated)
        if known_root_mean is not None:
            evaluated = tuple(u for u in evaluated if u != ROOT)

        steps = tuple(n + tree.depth_of(u) for u in evaluated)
        target_means = ops.zeros_stack(tree.size, root_state)
        if evaluated:
            ops.put(target_means, evaluated,
                    self.target((0,) * len(steps), ops.take(states, evaluated), steps))
        if known_root_mean is not None:
            ops.put(target_means, [ROOT], known_root_mean[None])
            if self._check_root_mean:
                self._verify_exact_root(ops, states, target_means, n)

        # -------------------------------------------------- Phase 3: acceptance
        # lines 15-23. Walk down from the root, stopping at the first rejection.
        u = ROOT
        committed = 0
        rejected = False
        examined: List[Optional[int]] = []
        for level in range(1, lookahead + 1):
            step = n + level - 1
            children = tree.children(u)
            request = VerifyRequest(
                step=step,
                proposal_mean=proposal_means[u],
                target_mean=target_means[u],
                sigma=self.schedule(step),
                children=ops.take(states, children),
                parent_state=states[u],
                rng=rng,
                backend=ops,
                info={"level": level, "node": u},
            )
            result: VerifyResult = self.verifier(request)
            ops.put(trajectory, [n + level], result.state[None])
            committed += 1
            examined.append(result.proposals_examined)

            # This parent's target mean was paid for during Phase 2 and is now
            # the freshest drift in existence; hand it to the proposal so a
            # delayed-drift scheme can prefetch it for the next round. Note this
            # happens whether or not the child was accepted -- the evaluation
            # was made either way, and on a rejection at level 1 it is the only
            # drift the next round will have.
            #
            # This is the *parent's* drift, one step behind the state the next
            # round will actually start from. `carry="nearest"` defers the hand-
            # off to the end of the round so it can pick a drift evaluated at
            # the committed step instead; see `_carry_nearest`.
            if self.prefetch == "parent":
                self._hand_over(ops, step, states[u], target_means[u])
            elif self.prefetch == "nearest":
                last_parent, last_children = u, children
                last_committed = result.state

            if not result.accepted:  # line 18: residual sample, round ends
                rejected = True
                break
            u = children[result.child_index]  # line 22

        if self.prefetch == "nearest" and committed:
            self._prefetch_nearest(
                ops, tree, n, states, target_means, has_mean,
                last_parent, last_children, last_committed, u, rejected,
            )

        return RoundRecord(
            start_step=n,
            lookahead=lookahead,
            committed=committed,  # bounded by lookahead: one per level, and the loop ends there
            accepted_depth=committed - 1 if rejected else committed,
            rejected=rejected,
            drafted=tree.budget,
            verified=len(evaluated),
            proposals_examined=tuple(examined),
        )

    def _verify_exact_root(self, ops, states, target_means, n: int) -> None:
        """Validate a reused root mean against a fresh evaluation.

        The optimisation rests on an invariant -- that the committed leaf of the
        previous round is this round's root at the same step. Truncation,
        topology changes, or indexing errors can violate that invariant.
        ``check_contract`` enables this validation. It calls ``target.means``
        directly, so the check does not increment the NFE counter.
        """
        fresh = self.target.means((0,), states[ROOT][None], (n,))[0]
        if not ops.allclose(fresh, target_means[ROOT]):
            raise ValueError(
                "the reused root target mean does not match a fresh evaluation "
                f"at step {n}. The exact-root optimisation assumed the previous "
                "round's committed leaf is this round's root; it is not."
            )

    def _prefetch_nearest(
        self, ops, tree, n, states, target_means, has_mean,
        parent, children, committed_state, terminal, rejected,
    ) -> None:
        """Hand the proposal the freshest drift available at the committed step.

        The delayed-drift proposal reuses one target drift for the next round.
        This method selects a drift evaluated at the committed step and, when
        possible, near the committed state.

        Three cases, in order of what is available:

        1.  **Full acceptance.** The round exhausted its lookahead, so the
            committed state is a leaf. With ``evaluate_leaves`` its own drift
            was computed in Phase 2 and is exact at the next root. Without it,
            leaves have no target mean and the method falls
            through to the parent.
        2.  **Rejection above the last level.** The committed state is a
            residual draw, not a drafted node, so no exact drift exists. Its
            siblings at that depth do sit at the same step and were verified in
            Phase 2, so carry the nearest one's. The residual is drawn close to
            the drafts by construction, making the nearest sibling a useful
            approximation.
        3.  **Rejection at the last level** (siblings are leaves, no
            ``evaluate_leaves``), or anything else unavailable: the parent's.

        All candidate drifts were evaluated in Phase 2, so this method adds no
        target evaluations.
        """
        depth = tree.depth_of(parent) + 1

        if not rejected and terminal in has_mean:
            # Case 1: the committed leaf's own drift, exact at the next root.
            # It is also the next root's target mean -- same state, same step --
            # so record it and let Phase 2 drop the root from its batch.
            self._hand_over(
                ops, n + tree.depth_of(terminal), states[terminal], target_means[terminal]
            )
            self._exact_root_mean = target_means[terminal]
            return

        usable = [v for v in children if v in has_mean]
        if rejected and usable:
            # Case 2: nearest sibling at the committed depth.
            best, best_d = usable[0], None
            for v in usable:
                d = ops.norm(states[v] - committed_state)
                if best_d is None or d < best_d:
                    best, best_d = v, d
            self._hand_over(ops, n + depth, states[best], target_means[best])
            return

        # Case 3: fall back to the parent, one step stale.
        self._hand_over(ops, n + depth - 1, states[parent], target_means[parent])


    def _hand_over(self, ops, step, state, target_mean) -> None:
        """Hand one verified node's drift to the proposal.

        The proposal interface is the same at any batch size, so a single-image
        run passes image 0 and stacks of one row.
        """
        self.proposal.on_verified(
            (0,), (step,), ops.stack_rows([state]), ops.stack_rows([target_mean])
        )


def standard_sampler(
    target: TargetTransition,
    schedule: NoiseSchedule,
    *,
    num_steps: int,
) -> SpeculativeSampler:
    """Reference (non-speculative) Euler-Maruyama loop, as a degenerate case.

    ``K = L = 1`` with a rule that always resamples: one committed step per
    target call, i.e. ``N`` NFEs. Useful as the denominator of every speedup
    number and as a distributional ground truth in tests.
    """
    from .kernels import IdentityProposal
    from .verify import ResampleVerifier

    return SpeculativeSampler(
        target=target,
        proposal=IdentityProposal(),
        schedule=schedule,
        tree=DraftTree.chain(1),
        verifier=ResampleVerifier(),
        num_steps=num_steps,
    )
