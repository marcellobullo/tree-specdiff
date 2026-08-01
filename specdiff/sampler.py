"""Algorithm 3: speculative diffusion sampling for an arbitrary draft tree.

The class below is a transcription of the pseudocode, with line numbers in the
comments. It owns the loop and the bookkeeping and nothing else -- topology
lives in :mod:`specdiff.trees`, the models in :mod:`specdiff.kernels`, and the
coupling in :mod:`specdiff.verify`.
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
        Wrap the verifier in :class:`~specdiff.verify.CheckedVerifier`. Cheap;
        leave it on while developing a new rule.

    Notes
    -----
    One instance samples one trajectory at a time. The parallelism the method
    exploits is *within* a round -- the whole draft tree goes through the
    target model in a single batched call -- which is a different axis from
    batching over images. Batching trajectories as well is possible but not
    free: independent trajectories accept different prefixes and so fall out of
    step, which needs either ragged batching or per-trajectory masking. See
    ``README.md``.
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
        backend: Optional[Backend] = None,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        verifier.check_topology(tree)
        self.target = target
        self.proposal = proposal
        self.schedule = schedule
        self.tree = tree
        self.verifier = CheckedVerifier(verifier) if check_contract else verifier
        self.num_steps = int(num_steps)
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
        self.proposal.reset()
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

        self.proposal.on_round_start(n, root_state)

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
            means = self.proposal.means(ops.take(states, parents), (step,) * len(parents))
            ops.put(proposal_means, parents, means)

            counts = [len(tree.children(u)) for u in parents]
            child_ids = [v for u in parents for v in tree.children(u)]
            noise = ops.randn_stack(len(child_ids), root_state, rng)
            ops.put(states, child_ids, ops.repeat_rows(means, counts) + sigma * noise)

        # ------------------------------------------------ Phase 2: verification
        # lines 11-13. The single target evaluation of the round, batched over
        # the internal nodes only: leaves are never parents, so their target
        # means are never needed (eq. 26: |I| = B / K).
        internal = tree.internal_nodes
        steps = tuple(n + tree.depth_of(u) for u in internal)
        target_means_batch = self.target(ops.take(states, internal), steps)
        target_means = ops.zeros_stack(tree.size, root_state)
        ops.put(target_means, internal, target_means_batch)

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
            self.proposal.on_verified(step, states[u], target_means[u])

            if not result.accepted:  # line 18: residual sample, round ends
                rejected = True
                break
            u = children[result.child_index]  # line 22

        return RoundRecord(
            start_step=n,
            lookahead=lookahead,
            committed=committed,  # bounded by lookahead: one per level, and the loop ends there
            accepted_depth=committed - 1 if rejected else committed,
            rejected=rejected,
            drafted=tree.budget,
            verified=len(internal),
            proposals_examined=tuple(examined),
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
