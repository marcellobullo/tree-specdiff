"""Transitions: the two mean maps and the shared variance schedule.

The model interface follows Appendix C, Equation 24::

    P_n(. | y) = N(m^p_n(y), sigma_n^2 I)
    Q_n(. | y) = N(m^q_n(y), sigma_n^2 I)

Any component that produces these means can be used, including a denoiser,
distilled draft network, analytic score, or delayed reverse drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

Array = Any


class NoiseSchedule(ABC):
    """The scale ``sigma_n`` shared by proposal and target at step ``n``.

    It is shared by assumption, not by convention: the rank-1 reduction of
    eqs. (8)-(11) is only valid when the two kernels differ solely in mean.
    """

    @abstractmethod
    def sigma(self, step: int) -> float: ...

    def __call__(self, step: int) -> float:
        s = float(self.sigma(step))
        if s <= 0.0:
            raise ValueError(
                f"sigma({step}) = {s}: speculation is vacuous at zero churn "
                "(the transitions become point masses, TV distance 1). "
                "See Remark 3 in the paper."
            )
        return s


class TabulatedSchedule(NoiseSchedule):
    """Pre-computed per-step scales, e.g. eq. (37) for the SD3 conversion."""

    def __init__(self, sigmas: Sequence[float]) -> None:
        self._sigmas = tuple(float(s) for s in sigmas)

    def sigma(self, step: int) -> float:
        return self._sigmas[step]

    def __len__(self) -> int:
        return len(self._sigmas)


class ConstantSchedule(NoiseSchedule):
    def __init__(self, sigma: float) -> None:
        self._sigma = float(sigma)

    def sigma(self, step: int) -> float:
        return self._sigma


class TargetTransition(ABC):
    """Expensive target mean map ``m^q``.

    Implementations override :meth:`means`. With refinement disabled it is
    called once with the round's verification batch; target-backed refinement
    adds one batched call per sweep and may eliminate the final call through
    exact cache reuse. Invoke the instance through ``__call__`` because it
    maintains NFE accounting.
    """

    def __init__(self) -> None:
        self.num_calls = 0
        self.num_states = 0

    @abstractmethod
    def means(
        self, indices_in_batch: Sequence[int], states: Array, steps: Sequence[int]
    ) -> Array:
        """``(rows, *state_shape)`` states -> means of the same shape.

        ``indices_in_batch[i]``, ``states[i]`` and ``steps[i]`` all describe
        entry ``i``: which of the ``batch_size`` images it belongs to, its
        value, and step. This matches :meth:`ProposalTransition.means`. Targets
        conditioned on per-image data, such as labels or prompts, use the
        index to select the corresponding condition. Unconditional targets may
        ignore it.

        Must be a single batched evaluation of the target network.
        Each output row must be deterministic for its image index, state, and
        step, and independent of other rows or the way the batch is partitioned.
        Exact verification and refinement target-cache reuse rely on this.
        """

    @abstractmethod
    def freeze_drift(self, states: Array, means: Array, steps: Sequence[int]) -> Array:
        """What :class:`DelayedDriftProposal` stores about a verified node.

        ``means`` are this target's means at ``(states, steps)``, one row per
        entry, and the return value is whatever per-row quantity the proposal
        should carry to the next round; :meth:`apply_drift` turns it back into
        a mean at a *different* state and step. Both must be exact inverses at
        a fixed ``(state, step)``.

        There is deliberately no default. Freeze the expensive quantity the
        mean is built from, not the mean itself: the churn kernels of the
        experiments are affine in the network velocity, ``m = a_n x + b_n v``,
        so they freeze ``v`` and re-apply the kernel at the drafted node's own
        state and step. Freezing the increment ``m^q(Y~) - Y~`` of eq. (7) and
        sliding it onto the drafted node would carry the score-correction term
        ``-(1/2) eps^2 g^2(sigma) score(x)`` evaluated at the stale state and
        step, an error that grows with ``eps``; that form is only right for a
        target whose mean is a translation of its input.
        """

    @abstractmethod
    def apply_drift(self, drift: Array, states: Array, steps: Sequence[int]) -> Array:
        """Rebuild a mean at ``(states, steps)`` from a frozen ``drift`` row.

        The inverse of :meth:`freeze_drift`; see there.
        """

    def __call__(
        self, indices_in_batch: Sequence[int], states: Array, steps: Sequence[int]
    ) -> Array:
        steps = tuple(int(s) for s in steps)
        indices_in_batch = tuple(int(b) for b in indices_in_batch)
        if len(indices_in_batch) != len(steps):
            raise ValueError(
                f"{len(indices_in_batch)} indices_in_batch for {len(steps)} steps; "
                "one per entry is required"
            )
        self.num_calls += 1
        self.num_states += len(steps)
        out = self.means(indices_in_batch, states, steps)
        if int(out.shape[0]) != len(steps):
            raise ValueError(
                f"{type(self).__name__}.means returned {int(out.shape[0])} rows "
                f"for {len(steps)} states"
            )
        return out

    def reset_stats(self) -> None:
        self.num_calls = 0
        self.num_states = 0


class ProposalTransition(ABC):
    """Proposal mean map ``m^p``, called once per drafted tree level.

    Every call carries ``indices_in_batch``: for each entry of the stack,
    which of the ``batch_size`` images it belongs to. Scalar sampling uses the
    same interface with ``batch_size = 1`` and zero-valued indices. Stateless
    proposals may ignore this argument; stateful proposals use it to isolate
    cached values by trajectory.

    The lifecycle hooks support proposals with per-trajectory state. Delayed
    reverse drift uses them to observe round boundaries and newly available
    target drifts. Stateless draft networks inherit the no-op implementations.
    """

    @abstractmethod
    def means(
        self, indices_in_batch: Sequence[int], states: Array, steps: Sequence[int]
    ) -> Array:
        """``(rows, *state_shape)`` states -> means of the same shape.

        ``indices_in_batch[i]``, ``states[i]`` and ``steps[i]`` all describe
        entry ``i``: which image it belongs to, its value, and its step.

        With JTX refinement, this base map is also evaluated at snapshot and
        rebuilt parents. It must be deterministic, row-local across batch
        partitions, and fixed throughout drafting and refinement in a round.
        ``means`` must not update proposal state; lifecycle hooks may do so
        between rounds.
        """

    def on_round_start(
        self, indices_in_batch: Sequence[int], steps: Sequence[int], roots: Array
    ) -> None:
        """Called once per round, before drafting, with each image's ``(n, Y_n)``."""

    def on_verified(
        self,
        indices_in_batch: Sequence[int],
        steps: Sequence[int],
        states: Array,
        target_means: Array,
    ) -> None:
        """Called for each *committed* node whose target mean was computed.

        This hook supports root-drift prefetching by exposing target drifts
        already evaluated during verification.
        """

    def configure_prefetch(self, mode: str) -> None:
        """Told by the sampler which prefetch policy is in force.

        ``mode`` is the sampler's ``prefetch`` setting: ``"none"``,
        ``"parent"`` or ``"nearest"``. A proposal that reuses a drift between
        rounds needs to know whether one will be supplied at all -- under
        ``"none"`` it must re-evaluate the target at every round's root, and
        under the others it must not. Proposals that keep no memory can ignore
        this; the default does nothing.
        """

    def configure_backend(self, backend) -> None:
        """Receive the sampler-selected backend for internal array operations."""

    def reset(self, batch_size: int) -> None:
        """Drop any cached memory and size it for ``batch_size`` images.

        Called at the start of :meth:`sample`.
        """


class IdentityProposal(ProposalTransition):
    """Zero-cost baseline proposal defined by ``m^p(y) = y``.

    Used by :func:`specdiff.sampler.standard_sampler`, where the drafts are
    discarded, and as a baseline for measuring proposal quality. It keeps no
    per-image memory, so ``indices_in_batch`` is
    ignored and one instance serves any batch size.
    """

    def means(self, indices_in_batch, states, steps):
        return states


class MirrorProposal(ProposalTransition):
    """Ideal diagnostic proposal defined by ``m^p = m^q``.

    It produces ``delta = 0`` and isolates sampler or verifier behavior from
    proposal error. It keeps no per-image memory.
    """

    def __init__(self, target: TargetTransition) -> None:
        self._target = target

    def means(self, indices_in_batch, states, steps):
        return self._target(indices_in_batch, states, steps)


class DelayedDriftProposal(ProposalTransition):
    """The self-speculative proposal of eq. (7), with root-drift prefetching.

    The target transition mean is ``m^q_n(y) = y + gamma b^q_{t_n}(y)``. The
    proposal freezes the drift behind a mean the sampler already computed at
    some node ``(Y~, n')`` and reuses it at every depth of the tree. In the
    paper's form that is the increment::

        m^p(y) = y + (m^q_{n'}(Y~) - Y~)

    What exactly is frozen, and how it is turned back into a mean at the
    drafted node, is the **target's** decision through
    :meth:`TargetTransition.freeze_drift` and
    :meth:`TargetTransition.apply_drift`; this class never forms the
    increment itself. The churn kernels of the experiments freeze the network
    velocity and re-run the step at the drafted node's state and step, which
    keeps the ``eps``-dependent score correction exact (see ``freeze_drift``).

    Because the drift is read from a target mean already computed during
    verification, no additional target call is needed per round (Appendix C,
    "root-drift prefetching"). One warm-up call per image is required at
    ``n = 0`` and included in the counters.

    ``_delayed_drift`` holds the frozen rows in a ``(batch_size, *state_shape)``
    buffer, one per image, indexed by ``indices_in_batch`` -- so drafting is a
    single gather-and-apply no matter how many images are in flight, no image
    can read another's drift, and the warm-up evaluations are collected into one
    target call rather than one per image.

    Parameters
    ----------
    target:
        Used for the warm-up calls, per round under ``prefetch="none"``, and
        for ``freeze_drift`` / ``apply_drift``.
    Which drift gets reused is the **sampler's** ``prefetch`` setting, not this
    class's: selecting it needs the tree and the drafted states, which a
    proposal cannot see. The sampler announces the policy through
    :meth:`configure_prefetch` and then hands over the chosen drift through
    :meth:`on_verified`; all this class does is hold it and apply it.
    """

    def __init__(self, target: TargetTransition) -> None:
        self._target = target
        self._refresh_each_round = False
        self._delayed_drift: Optional[Array] = None
        self._have: list[bool] = []
        self._batch_size = 0
        self._backend = None

    def configure_backend(self, backend) -> None:
        self._backend = backend

    def reset(self, batch_size: int) -> None:
        self._delayed_drift = None
        self._have = [False] * batch_size
        self._batch_size = batch_size

    def configure_prefetch(self, mode: str) -> None:
        # "none" means no drift is handed over, so the root must be paid for.
        self._refresh_each_round = mode == "none"

    def on_round_start(self, indices_in_batch, steps, roots) -> None:
        from .ops import resolve_backend

        ops = self._backend or resolve_backend(roots)
        if self._delayed_drift is None:
            self._delayed_drift = ops.zeros_stack(self._batch_size, roots[0])
        missing = [
            i for i, b in enumerate(indices_in_batch)
            if self._refresh_each_round or not self._have[b]
        ]
        if not missing:
            return
        rows = ops.take(roots, missing)
        missing_steps = tuple(steps[i] for i in missing)
        means = self._target([indices_in_batch[i] for i in missing], rows, missing_steps)
        ops.put(
            self._delayed_drift,
            [indices_in_batch[i] for i in missing],
            self._target.freeze_drift(rows, means, missing_steps),
        )
        for i in missing:
            self._have[indices_in_batch[i]] = True

    def on_verified(self, indices_in_batch, steps, states, target_means) -> None:
        if self._refresh_each_round:
            return
        from .ops import resolve_backend

        ops = self._backend or resolve_backend(states)
        ops.put(
            self._delayed_drift,
            list(indices_in_batch),
            self._target.freeze_drift(states, target_means, tuple(steps)),
        )
        for b in indices_in_batch:
            self._have[b] = True

    def means(self, indices_in_batch, states, steps):
        from .ops import resolve_backend

        if self._delayed_drift is None:
            raise RuntimeError("on_round_start must run before drafting")
        ops = self._backend or resolve_backend(states)
        drift = ops.take(self._delayed_drift, list(indices_in_batch))
        return self._target.apply_drift(drift, states, tuple(steps))
