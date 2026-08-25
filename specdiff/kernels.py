"""Transitions: the two mean maps and the shared variance schedule.

Appendix C, eq. (24) is the entire model interface::

    P_n(. | y) = N(m^p_n(y), sigma_n^2 I)
    Q_n(. | y) = N(m^q_n(y), sigma_n^2 I)

Anything that can produce those means plugs in: a real denoiser, a distilled
draft network, an analytic score, or the paper's delayed reverse drift. The
sampler never learns which.
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
    """``m^q``. Evaluating this is the expensive thing we are trying to avoid.

    Implementations override :meth:`means`, which is called **once per round**
    with the whole verification batch (all internal nodes of the round's tree).
    Call the instance, don't call ``means`` directly: ``__call__`` keeps the
    NFE accounting that the cost metric is defined on.
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
        value, and its step. Same convention as
        :meth:`ProposalTransition.means`, and for the same reason: one call
        carries entries from several images, so a target that conditions on
        anything per-image -- a class label, a text prompt -- needs to know
        which is which. A target that conditions on nothing ignores it.

        Must be a single batched evaluation of the target network.
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
    """``m^p``. Cheap by assumption; called once per tree level while drafting.

    Every call carries ``indices_in_batch``: for each entry of the stack,
    which of the ``batch_size`` images it belongs to. There is one interface,
    not two -- sampling a single image is ``batch_size = 1``, where
    ``indices_in_batch`` is all zeros, and not a separate world with its own
    class hierarchy. A proposal that keeps no per-image memory simply ignores
    the argument.

    That uniformity is why no adapter classes exist. Lifting a one-image
    proposal into a batch used to need a wrapper, and a wrapper that shared one
    object across images would silently apply image 0's memory to image 3 --
    so a second wrapper existed to make one object per image, and a flag to
    say which wrapper you needed. None of that is reachable now: a proposal is
    told which image each entry belongs to, so it can always do the right
    thing in one call.

    The three hooks exist because the interesting proposals carry memory. A
    delayed reverse drift has to be told when a round starts and which target
    drifts have become available; a distilled draft network needs neither and
    inherits the no-ops.
    """

    @abstractmethod
    def means(
        self, indices_in_batch: Sequence[int], states: Array, steps: Sequence[int]
    ) -> Array:
        """``(rows, *state_shape)`` states -> means of the same shape.

        ``indices_in_batch[i]``, ``states[i]`` and ``steps[i]`` all describe
        entry ``i``: which image it belongs to, its value, and its step.
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

        This is the channel that makes root-drift prefetching possible: the
        drift the proposal will reuse next round is one this round already paid
        for during verification.
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

    def reset(self, batch_size: int) -> None:
        """Drop any cached memory and size it for ``batch_size`` images.

        Called at the start of :meth:`sample`.
        """


class IdentityProposal(ProposalTransition):
    """``m^p(y) = y``: the worst useful proposal, and free.

    Used by :func:`specdiff.sampler.standard_sampler`, where the drafts are
    discarded anyway, and as a floor when measuring how much proposal quality
    is buying you. Keeps no per-image memory, so ``indices_in_batch`` is
    ignored and one instance serves any batch size.
    """

    def means(self, indices_in_batch, states, steps):
        return states


class MirrorProposal(ProposalTransition):
    """``m^p = m^q``: a perfect proposal (``delta = 0``, everything accepts).

    Useless in production, invaluable in tests: it isolates bugs in a
    verification rule from bugs in the coupling, and it makes the sampler's
    accounting easy to reason about. Keeps no per-image memory.
    """

    def __init__(self, target: TargetTransition) -> None:
        self._target = target

    def means(self, indices_in_batch, states, steps):
        return self._target(indices_in_batch, states, steps)


class DelayedDriftProposal(ProposalTransition):
    """The self-speculative proposal of eq. (7), with root-drift prefetching.

    The target transition mean is ``m^q_n(y) = y + gamma b^q_{t_n}(y)``, so the
    increment ``gamma b^q`` can be recovered from means alone::

        gamma b^q_{t_n'}(Y~) = m^q_{n'}(Y~) - Y~

    The proposal then freezes that increment and reuses it at every depth of
    the tree::

        m^p(y) = y + (m^q_{n'}(Y~) - Y~)

    Because the increment is read off a target mean that verification already
    computed in an earlier round, no extra target call is needed per round
    (Appendix C, "root-drift prefetching"). Exactly one warm-up call per image
    is unavoidable at ``n = 0``, and it is counted.

    ``_delayed_drift`` holds those increments in a ``(batch_size, *state_shape)``
    buffer, one row per image, indexed by ``indices_in_batch`` -- so drafting is
    a single gather-and-add no matter how many images are in flight, no image
    can read another's drift, and the warm-up evaluations are collected into one
    target call rather than one per image.

    Parameters
    ----------
    target:
        Used for the warm-up calls, and per round under ``prefetch="none"``.
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

    def reset(self, batch_size: int) -> None:
        self._delayed_drift = None
        self._have = [False] * batch_size
        self._batch_size = batch_size

    def configure_prefetch(self, mode: str) -> None:
        # "none" means no drift is handed over, so the root must be paid for.
        self._refresh_each_round = mode == "none"

    def on_round_start(self, indices_in_batch, steps, roots) -> None:
        from .ops import resolve_backend

        ops = resolve_backend(roots)
        if self._delayed_drift is None:
            self._delayed_drift = ops.zeros_stack(self._batch_size, roots[0])
        missing = [
            i for i, b in enumerate(indices_in_batch)
            if self._refresh_each_round or not self._have[b]
        ]
        if not missing:
            return
        rows = ops.take(roots, missing)
        means = self._target(
            [indices_in_batch[i] for i in missing], rows, tuple(steps[i] for i in missing)
        )
        ops.put(self._delayed_drift, [indices_in_batch[i] for i in missing], means - rows)
        for i in missing:
            self._have[indices_in_batch[i]] = True

    def on_verified(self, indices_in_batch, steps, states, target_means) -> None:
        if self._refresh_each_round:
            return
        from .ops import resolve_backend

        ops = resolve_backend(states)
        ops.put(self._delayed_drift, list(indices_in_batch), target_means - states)
        for b in indices_in_batch:
            self._have[b] = True

    def means(self, indices_in_batch, states, steps):
        from .ops import resolve_backend

        if self._delayed_drift is None:
            raise RuntimeError("on_round_start must run before drafting")
        ops = resolve_backend(states)
        return states + ops.take(self._delayed_drift, list(indices_in_batch))
