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
    def means(self, states: Array, steps: Sequence[int]) -> Array:
        """``(rows, *state_shape)`` states at ``rows`` step indices -> means of the
        same shape. ``rows`` is however many nodes the caller batched together.

        Must be a single batched evaluation of the target network.
        """

    def __call__(self, states: Array, steps: Sequence[int]) -> Array:
        steps = tuple(int(s) for s in steps)
        self.num_calls += 1
        self.num_states += len(steps)
        out = self.means(states, steps)
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

    The three hooks exist because the interesting proposals are stateful. A
    delayed reverse drift has to be told when a round starts and which target
    drifts have become available; a distilled draft network needs neither and
    inherits the no-ops.

    Subclasses that keep state across calls must set ``stateful = True``. It is
    what stops a single-trajectory proposal from being silently reused across a
    batch, where one cached drift would be shared by trajectories sitting at
    different steps. See :mod:`specdiff.batched`.
    """

    stateful: bool = False

    @abstractmethod
    def means(self, states: Array, steps: Sequence[int]) -> Array: ...

    def on_round_start(self, step: int, root_state: Array) -> None:
        """Called once per round, before drafting, with ``(n, Y_n)``."""

    def on_verified(self, step: int, state: Array, target_mean: Array) -> None:
        """Called for each *committed* node whose target mean was computed.

        This is the channel that makes root-drift prefetching possible: the
        drift the proposal will reuse next round is one this round already paid
        for during verification.
        """

    def reset(self) -> None:
        """Drop any cached state. Called at the start of :meth:`sample`."""


class IdentityProposal(ProposalTransition):
    """``m^p(y) = y``: the worst useful proposal, and free.

    Used by :func:`specdiff.sampler.standard_sampler`, where the drafts are
    discarded anyway, and as a floor when measuring how much proposal quality
    is buying you.
    """

    def means(self, states, steps):
        return states


class MirrorProposal(ProposalTransition):
    """``m^p = m^q``: a perfect proposal (``delta = 0``, everything accepts).

    Useless in production, invaluable in tests: it isolates bugs in a
    verification rule from bugs in the coupling, and it makes the driver's
    accounting easy to reason about.
    """

    def __init__(self, target: TargetTransition) -> None:
        self._target = target

    def means(self, states, steps):
        return self._target(states, steps)


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
    (Appendix C, "root-drift prefetching"). Exactly one warm-up call is
    unavoidable at ``n = 0``, and it is counted.

    Parameters
    ----------
    target:
        Used only for the warm-up call, and per round if ``prefetch=False``.
    prefetch:
        ``True`` reuses the freshest committed drift (the paper's default).
        ``False`` re-evaluates the target at the root of every round, which
        costs one extra NFE per round but gives a strictly better proposal --
        useful for isolating the effect of proposal quality.
    """

    stateful = True

    def __init__(self, target: TargetTransition, *, prefetch: bool = True) -> None:
        self._target = target
        self._prefetch = prefetch
        self._increment: Optional[Array] = None

    def reset(self) -> None:
        self._increment = None

    def on_round_start(self, step: int, root_state: Array) -> None:
        if self._increment is None or not self._prefetch:
            mean = self._target(root_state[None], (step,))[0]
            self._increment = mean - root_state

    def on_verified(self, step: int, state: Array, target_mean: Array) -> None:
        if self._prefetch:
            self._increment = target_mean - state

    def means(self, states, steps):
        if self._increment is None:
            raise RuntimeError("on_round_start must run before drafting")
        return states + self._increment
