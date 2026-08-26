"""Data types for the verifier extension boundary.

Appendix C defines the verifier contract as follows:

    Given a parent node ``u``, its associated proposal and target means, the
    corresponding step scale, and its set of drafted children, the rule outputs
    a tuple ``(Y, accepted, v*)``. ``Y`` is an exact sample from
    ``Q(. | Y_u)``; either ``accepted`` is true and ``Y`` is one of the
    children, or ``accepted`` is false and ``Y`` came from the residual.

:class:`VerifyRequest` represents the inputs, and :class:`VerifyResult`
represents the output. A verification rule maps between these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

Array = Any


@dataclass(frozen=True)
class VerifyRequest:
    """Inputs available to a verification rule at one node.

    The template assumes (Appendix C, eq. 24) that proposal and target
    transitions are isotropic Gaussians that *share* the variance schedule and
    differ only in their means::

        P_step(. | parent_state) = N(proposal_mean, sigma^2 I)
        Q_step(. | parent_state) = N(target_mean,   sigma^2 I)

    This type encodes the shared-covariance assumption used by the rank-1
    reduction in Equations 8--11, so verifiers may rely on it.

    Attributes
    ----------
    step:
        Index ``n + |u|`` of the transition being verified, i.e. the step the
        *children* realise minus one. Rules that are time-dependent (adaptive
        budgets, annealed thresholds) can key off it.
    parent_state:
        ``Y_u``. Not needed by RMC or D-GRS, but a rule that wants to re-run the
        proposal or inspect the trajectory should not have to be re-plumbed.
    proposal_mean, target_mean, sigma:
        ``m^p_step(Y_u)``, ``m^q_step(Y_u)`` and ``sigma_step``.
    children:
        Stack of shape ``(K, *state_shape)`` holding the drafted children
        ``{Y_v : v in C(u)}``. **The row order is the order in which the
        children were sampled** and is the order a sequence-coupling rule must
        examine them in; the sampler guarantees it is stable.
    rng:
        Backend random generator, or ``None`` for the global stream. Rules must
        use this rather than a private one, so that a run is reproducible.
    info:
        Free-form channel for extras a custom rule needs (guidance scale,
        per-node budget, ...). Empty by default; the sampler never reads it.
    """

    step: int
    proposal_mean: Array
    target_mean: Array
    sigma: float
    children: Array
    parent_state: Optional[Array] = None
    index_in_batch: int = 0
    """Which trajectory this node belongs to. Always ``0`` under the
    single-trajectory sampler; under :mod:`specdiff.batched` it identifies the
    trajectory, which a rule holding per-trajectory state needs in order to
    behave the same way whether it is called row-wise or in a batch."""
    rng: Any = None
    backend: Any = None
    """Array backend selected by the sampler. Custom backends are carried
    explicitly because their array types are not known to ``resolve_backend``."""
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_children(self) -> int:
        """K, the branching factor at this node."""
        return int(self.children.shape[0])

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return tuple(self.children.shape[1:])

    def child(self, index: int) -> Array:
        return self.children[index]


@dataclass(frozen=True)
class VerifyResult:
    """The tuple ``(Y, accepted, v*)``, plus optional telemetry.

    Parameters
    ----------
    state:
        ``Y``, which **must** be an exact sample from
        ``N(request.target_mean, request.sigma^2 I)``. This is the one
        obligation the library cannot check at runtime; see
        :func:`specdiff.testing.check_exactness` for a statistical test to run
        in the verifier's test suite.
    accepted:
        Whether ``state`` is one of the drafted children.
    child_index:
        Row of ``request.children`` that was accepted; required iff
        ``accepted``. The sampler maps it back to the tree node ``v*``.
    proposals_examined:
        How many children the rule looked at (the third return value of
        Algorithms 1 and 2: ``1`` for RMC, ``k`` or ``K + 1`` for GRS). Purely
        diagnostic; ``None`` if the rule does not track it.
    """

    state: Array
    accepted: bool
    child_index: Optional[int] = None
    proposals_examined: Optional[int] = None

    def __post_init__(self) -> None:
        if self.accepted and self.child_index is None:
            raise ValueError("accepted=True requires child_index (the accepted v*)")
        if not self.accepted and self.child_index is not None:
            raise ValueError("child_index must be None when accepted=False")


@dataclass(frozen=True)
class RoundRecord:
    """Diagnostics for one iteration of the outer loop of Algorithm 3."""

    start_step: int
    """``n`` at the top of the round."""
    lookahead: int
    """``L_n = min(L, N - n)``."""
    committed: int
    """``N_alpha``: states committed this round, always in ``[1, L_n]``."""
    accepted_depth: int
    """Length of the accepted prefix, i.e. ``committed - 1`` if the round ended
    in a rejection and ``committed`` if every level was accepted."""
    rejected: bool
    drafted: int
    """Number of drafted states ``B_n = |V(T_n)| - 1``."""
    verified: int
    """Size of the verification batch, ``|I(T_n)|``. One target call."""
    proposals_examined: Tuple[Optional[int], ...] = ()

    @property
    def steps_per_target_call(self) -> float:
        return float(self.committed)


@dataclass
class SamplingResult:
    """Output of :meth:`specdiff.sampler.SpeculativeSampler.sample`."""

    trajectory: Array
    """Stack of shape ``(N + 1, *state_shape)``: ``Y_0`` through ``Y_N``."""
    rounds: Tuple[RoundRecord, ...]
    num_steps: int
    target_calls: int
    """Neural function evaluations, the paper's cost metric: one batched call
    per round, plus any warm-up call a proposal needed."""
    target_states_evaluated: int
    """Total states pushed through the target model (batch volume, not NFEs)."""
    drafted_states: int

    @property
    def sample(self) -> Array:
        return self.trajectory[-1]

    @property
    def speedup(self) -> float:
        """NFEs of the standard sampler divided by NFEs actually spent."""
        return self.num_steps / max(self.target_calls, 1)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of verified levels that accepted a drafted child."""
        levels = sum(r.committed for r in self.rounds)
        accepted = sum(r.accepted_depth for r in self.rounds)
        return accepted / max(levels, 1)

    def summary(self) -> str:
        return (
            f"steps={self.num_steps} target_calls={self.target_calls} "
            f"speedup={self.speedup:.3f}x acceptance={self.acceptance_rate:.3f} "
            f"drafted={self.drafted_states} verified={self.target_states_evaluated}"
        )


@dataclass(frozen=True)
class BatchedVerifyRequest:
    """``batch`` independent nodes, one per live trajectory, verified together.

    This is :class:`VerifyRequest` with a leading batch dimension, and one
    substantive difference: ``sigmas`` is per-row, not scalar. Trajectories in
    a batch accept different prefixes, so after the first round they sit at
    different steps ``n_i`` and no longer share a noise scale. Any rule that
    wants to vectorise must broadcast over it -- ``ops.scale_rows`` is the
    portable way.

    All rows are live: finished or already-rejected trajectories are dropped
    from the batch by the sampler rather than masked, so a rule never has to
    reason about validity. ``indices_in_batch`` says which image each row
    belongs to, for rules that keep per-image memory.

    ``info`` mirrors the scalar sampler's, with one difference forced by the
    batch: the scalar request carries the tree node as ``info["node"]``, but
    rows sit at different nodes, so the batched request carries the whole tuple
    as ``info["nodes"]``. :meth:`row` translates it back, so a rule written
    against the scalar contract sees ``info["node"]`` either way.
    """

    steps: Tuple[int, ...]
    indices_in_batch: Tuple[int, ...]
    proposal_mean: Array  # (batch, *shape)
    target_mean: Array  # (batch, *shape)
    sigmas: Tuple[float, ...]
    children: Array  # (batch, K, *shape)
    parent_state: Optional[Array] = None
    rng: Any = None
    backend: Any = None
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return len(self.steps)

    @property
    def num_children(self) -> int:
        return int(self.children.shape[1])

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return tuple(self.children.shape[2:])

    def row(self, j: int) -> VerifyRequest:
        """Extract row ``j`` as a single-node request.

        The resulting ``info`` matches the scalar sampler's exactly: the
        batch-wide ``"nodes"`` tuple becomes this row's ``"node"``.
        """
        info = self.info
        nodes = info.get("nodes")
        if nodes is not None:
            info = {k: v for k, v in info.items() if k != "nodes"}
            info["node"] = nodes[j]
        return VerifyRequest(
            step=self.steps[j],
            proposal_mean=self.proposal_mean[j],
            target_mean=self.target_mean[j],
            sigma=self.sigmas[j],
            children=self.children[j],
            parent_state=None if self.parent_state is None else self.parent_state[j],
            index_in_batch=self.indices_in_batch[j],
            rng=self.rng,
            backend=self.backend,
            info=info,
        )


@dataclass(frozen=True)
class BatchedVerifyResult:
    """Per-row outcomes. Same contract as :class:`VerifyResult`, row by row."""

    states: Array  # (batch, *shape)
    accepted: Tuple[bool, ...]
    child_index: Tuple[Optional[int], ...]
    proposals_examined: Tuple[Optional[int], ...] = ()

    def __post_init__(self) -> None:
        if len(self.accepted) != len(self.child_index):
            raise ValueError("accepted and child_index must have the same length")
        if int(self.states.shape[0]) != len(self.accepted):
            raise ValueError(
                f"states has {int(self.states.shape[0])} rows but {len(self.accepted)} flags"
            )
        for ok, idx in zip(self.accepted, self.child_index):
            if ok and idx is None:
                raise ValueError("an accepted row needs a child_index")
            if not ok and idx is not None:
                raise ValueError("a rejected row must not carry a child_index")

    @classmethod
    def from_rows(cls, results: Sequence[VerifyResult], ops) -> "BatchedVerifyResult":
        return cls(
            states=ops.stack_rows([r.state for r in results]),
            accepted=tuple(bool(r.accepted) for r in results),
            child_index=tuple(r.child_index for r in results),
            proposals_examined=tuple(r.proposals_examined for r in results),
        )


@dataclass(frozen=True)
class BatchedRoundRecord:
    """One outer iteration of the batched sampler: a single target call."""

    iteration: int
    active: Tuple[int, ...]
    """Images that took part, by index in the batch."""
    start_steps: Tuple[int, ...]
    committed: Tuple[int, ...]
    """States committed per active trajectory, all >= 1. Equivalently, levels
    of the tree this trajectory had verified before its round ended."""
    drafted: int
    verified: int
    """Rows in this iteration's single batched target call."""
    accepted_depth: Tuple[int, ...] = ()
    """Accepted prefix length per active trajectory: ``committed`` if every
    verified level accepted, ``committed - 1`` if the round ended in a
    rejection. Recorded because acceptance rate is the paper's headline
    diagnostic and it cannot be recovered from ``committed`` alone."""
    rejected: Tuple[bool, ...] = ()
    """Whether each active trajectory's round ended in a rejection."""


@dataclass
class BatchedSamplingResult:
    """Output of :meth:`specdiff.batched.BatchedSpeculativeSampler.sample`."""

    samples: Array
    """Terminal states, ``(batch, *state_shape)``."""
    trajectories: Optional[Array]
    """``(batch, N + 1, *state_shape)`` if ``keep_trajectories``, else ``None``."""
    rounds: Tuple[BatchedRoundRecord, ...]
    num_steps: int
    batch_size: int
    target_calls: int
    """Batched target calls, i.e. NFEs of wall-clock depth."""
    target_states_evaluated: int
    drafted_states: int
    rounds_per_trajectory: Tuple[int, ...]

    @property
    def speedup(self) -> float:
        """Wall-clock speedup of the batch: ``N / target_calls``.

        This is the number that matters when you generate a batch, and it is
        *not* the mean of the per-trajectory speedups. One batched call serves
        every live trajectory, so the batch advances at the pace of its
        slowest member; see :attr:`mean_isolated_speedup`.
        """
        return self.num_steps / max(self.target_calls, 1)

    @property
    def mean_isolated_speedup(self) -> float:
        """Mean over trajectories of ``N / rounds_i``: what each would have
        achieved run on its own. The gap to :attr:`speedup` is the straggler
        cost of sharing a batch."""
        return sum(self.num_steps / max(r, 1) for r in self.rounds_per_trajectory) / max(
            self.batch_size, 1
        )

    @property
    def occupancy(self) -> float:
        """Mean fraction of the batch still live per iteration."""
        if not self.rounds:
            return 0.0
        return sum(len(r.active) for r in self.rounds) / (len(self.rounds) * self.batch_size)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of verified levels that accepted a drafted child.

        Pooled over every trajectory and every round, so it is directly
        comparable with :attr:`specdiff.types.SamplingResult.acceptance_rate`
        from a single-trajectory run of the same configuration.
        """
        levels = sum(sum(r.committed) for r in self.rounds)
        accepted = sum(sum(r.accepted_depth) for r in self.rounds)
        return accepted / max(levels, 1)

    def summary(self) -> str:
        return (
            f"steps={self.num_steps} batch={self.batch_size} "
            f"target_calls={self.target_calls} speedup={self.speedup:.3f}x "
            f"(isolated {self.mean_isolated_speedup:.3f}x, occupancy {self.occupancy:.2f}) "
            f"acceptance={self.acceptance_rate:.3f} "
            f"drafted={self.drafted_states} verified={self.target_states_evaluated}"
        )
