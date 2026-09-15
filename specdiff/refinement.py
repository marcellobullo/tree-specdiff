"""Synchronous Picard refinement for drafted diffusion trees.

The sampler owns topology and buffer layout. This module evaluates one
row-local update for every internal node from a frozen snapshot, then
reconstructs all children breadth-first using fixed edge innovations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Callable, Optional, Tuple

from .kernels import ProposalTransition, TargetTransition
from .ops import Backend

Array = Any
BroydenFactors = Tuple[Tuple[Array, Array], ...]


@dataclass(frozen=True)
class RefinementRequest:
    """Inputs to one synchronous refinement sweep.

    Output row ``i`` must depend only on input row ``i``, its own history
    from preceding sweeps, fixed model parameters, and conditioning for
    ``indices_in_batch[i]``. Mixing rows,
    including through cross-row reductions, can make a parent's proposal mean
    depend on its own outgoing noise and invalidates conditional Gaussianity.
    """

    iteration: int
    indices_in_batch: Tuple[int, ...]
    nodes: Tuple[int, ...]
    steps: Tuple[int, ...]
    parent_states: Array
    current_proposal_means: Array
    sigmas: Tuple[float, ...]
    target: TargetTransition
    backend: Backend
    proposal: Optional[ProposalTransition] = None
    # Shared across sweeps of this round only. History must remain row-local.
    history: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class RefinementUpdate:
    """A row-local update and optional correctness-bearing target cache.

    Set exactly one of ``increments``, ``drifts``, and
    ``base_proposal_means``, each shaped like
    ``request.parent_states``. With ``X_u`` the parent state rebuilt earlier in
    the same sweep, the new proposal mean is ``X_u + increments[i]``, or
    ``target.apply_drift(drifts[i], X_u, step)``. ``apply_drift`` must then be
    row-local too; when ``exact_target_means`` is also supplied, the drift mean
    is anchored on it so that unmoved parents keep the exact target mean.

    ``base_proposal_means`` selects JTX mean-error transport and requires
    ``exact_target_means``. It stores the base proposal at the snapshot, not
    the current corrected mean. Reconstruction uses ``q(snapshot) +
    (p(rebuilt_parent) - p(snapshot))`` with the round's fixed proposal map.

    Optional ``broyden_factors[i]`` adds ``B_i (rebuilt_parent - snapshot)``
    to the JTX mean, with ``B_i = sum(u v.T)`` over that row's factor pairs.
    Factors have the shape of a single state, never mix rows, and must remain
    unchanged while the update is reconstructed.

    If supplied, ``exact_target_means[i]`` asserts that it is exactly the
    target mean at ``request.parent_states[i]`` for the associated image and
    step. A false assertion can invalidate exact sampling in the same way as
    an incorrect custom verifier. ``check_contract=True`` validates reused
    cache rows.
    """

    increments: Optional[Array] = None
    exact_target_means: Optional[Array] = None
    drifts: Optional[Array] = None
    base_proposal_means: Optional[Array] = None
    broyden_factors: Optional[Tuple[BroydenFactors, ...]] = None

    def __post_init__(self) -> None:
        if sum(value is not None for value in (
            self.increments, self.drifts, self.base_proposal_means
        )) != 1:
            raise TypeError(
                "RefinementUpdate needs exactly one of increments, drifts, and base_proposal_means"
            )
        if self.base_proposal_means is not None and self.exact_target_means is None:
            raise TypeError("base_proposal_means requires exact_target_means")
        if self.broyden_factors is not None and self.base_proposal_means is None:
            raise TypeError("broyden_factors requires base_proposal_means")


RefinementUpdateFn = Callable[[RefinementRequest], RefinementUpdate]


@dataclass(frozen=True)
class RefinementLevel:
    """One breadth-first reconstruction level in flat-buffer coordinates."""

    parent_ids: Tuple[int, ...]
    parent_positions: Tuple[int, ...]
    child_ids: Tuple[int, ...]
    child_counts: Tuple[int, ...]
    child_innovation_ids: Tuple[int, ...]


@dataclass(frozen=True)
class RefinementLayout:
    """Sampler-provided description of logical nodes in flat state buffers."""

    root_ids: Tuple[int, ...]
    internal_ids: Tuple[int, ...]
    logical_nodes: Tuple[int, ...]
    indices_in_batch: Tuple[int, ...]
    steps: Tuple[int, ...]
    sigmas: Tuple[float, ...]
    levels: Tuple[RefinementLevel, ...]

    def __post_init__(self) -> None:
        rows = len(self.internal_ids)
        fields = (self.logical_nodes, self.indices_in_batch, self.steps, self.sigmas)
        if any(len(field) != rows for field in fields):
            raise ValueError("refinement layout fields must have one entry per internal node")


@dataclass(frozen=True)
class ExactTargetCache:
    """Exact target evaluations made during the last refinement sweep."""

    target: TargetTransition
    flat_ids: Tuple[int, ...]
    indices_in_batch: Tuple[int, ...]
    nodes: Tuple[int, ...]
    steps: Tuple[int, ...]
    states: Array
    means: Array


@dataclass(frozen=True)
class ExactTargetMean:
    """One exact target value carried across sampler rounds."""

    target: TargetTransition
    index_in_batch: int
    step: int
    state: Array
    mean: Array


def normalize_refinement_iters(value: Optional[int]) -> int:
    """Validate the public ``proposal_refinement_iters`` argument."""

    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("proposal_refinement_iters must be None or a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError("proposal_refinement_iters must be >= 0")
    return value


def picard_update_fn(request: RefinementRequest) -> RefinementUpdate:
    """Target-backed Picard update that freezes the whole increment ``m^q_s(x) - x``.

    On a chain, this is the reference ParaDiGMS prefix-sum recurrence for
    matching transition means, initial guesses, and fixed noises, in exact
    arithmetic. It was the default before ``picard_drift_update_fn`` and
    preserves the update used by earlier runs. Neither split universally
    dominates the other; see docs/refinement.md for a cancellation example.
    """

    means = request.target(
        request.indices_in_batch,
        request.parent_states,
        request.steps,
    )
    return RefinementUpdate(
        increments=means - request.parent_states,
        exact_target_means=means,
    )


def picard_drift_update_fn(request: RefinementRequest) -> RefinementUpdate:
    """Default target-backed Picard update: freeze only the target's drift.

    ``picard_update_fn`` freezes the whole increment ``m^q_s(x) - x`` at the
    snapshot. This update freezes ``target.freeze_drift`` of the same mean, and
    ``refine_tree`` re-applies it with ``target.apply_drift`` at the rebuilt
    parent. For the churn kernels, ``m = a x + b v``, the affine part ``a x``
    is then evaluated at the new state and only the network velocity is
    frozen. That removes the stale ``(a - 1)(X^(j+1) - X^(j))`` term from the
    proposal mean. Both updates have the same fixed point, and they agree up to
    rounding for a target whose frozen drift is the increment itself. See
    ``docs/refinement.md`` for the derivations.
    """

    means = request.target(
        request.indices_in_batch,
        request.parent_states,
        request.steps,
    )
    return RefinementUpdate(
        drifts=request.target.freeze_drift(request.parent_states, means, request.steps),
        exact_target_means=means,
    )


def picard_jtx_update_fn(request: RefinementRequest) -> RefinementUpdate:
    """Transport the target-minus-base-proposal error (JTX, PDF eqs. 3–4).

    Freeze ``q(old) - p(old)`` and rebuild with ``p(new)``. The base proposal
    must be deterministic, row-local, and fixed throughout drafting and all
    sweeps in a round. In particular, the stored corrected proposal means
    from the previous sweep are not the base ``p(old)``.

    Keep both snapshot means so reconstruction can use
    ``q(old) + (p(new) - p(old))``. Unmoved parents then retain the exact
    target mean bit for bit. One target batch and one base-proposal batch
    are evaluated here; rebuilding evaluates the base proposal per depth.
    """

    if request.proposal is None:
        raise ValueError("picard_jtx_update_fn requires the base proposal")
    base_means = request.proposal.means(
        request.indices_in_batch, request.parent_states, request.steps
    )
    means = request.target(
        request.indices_in_batch, request.parent_states, request.steps
    )
    return RefinementUpdate(base_proposal_means=base_means, exact_target_means=means)


@dataclass(frozen=True)
class _BroydenHistory:
    iteration: int
    state: Array
    error: Array
    factors: BroydenFactors


def _apply_broyden(factors: BroydenFactors, displacement: Array, ops: Backend) -> Array:
    result = displacement * 0
    for u, v in factors:
        result = result + u * ops.dot(v, displacement)
    return result


def picard_broyden_correction_update_fn(
    request: RefinementRequest, *, memory: int = 2,
) -> RefinementUpdate:
    """JTX with limited-memory, per-parent good-Broyden corrections.

    For ``d=q-p``, successive snapshots supply ``s=x_j-x_(j-1)`` and
    ``z=d(x_j)-d(x_(j-1))``. Starting from B=0, fit
    ``B <- B + (z-B s) s.T / (s.T s)`` and transport with
    ``q(old) + (p(new)-p(old)) + B (new-old)``.

    Retain at most ``memory`` rank-one factors per (image, node, step),
    dropping the oldest *before* fitting the newest secant. Normalized
    directions avoid explicitly squaring tiny displacements. Skip movements
    no larger than sqrt(machine epsilon) times max(1, ||old||, ||new||),
    and non-finite updates. Non-finite transported means fall back to JTX.

    ``request.history`` is owned by refine_tree and discarded each round.
    No global callback state or extra target queries are used. Memory zero
    is bitwise JTX. Use functools.partial to configure a different memory.
    """
    if isinstance(memory, bool) or not isinstance(memory, Integral):
        raise TypeError("Broyden memory must be a non-negative integer")
    if memory < 0:
        raise ValueError("Broyden memory must be >= 0")
    memory = int(memory)
    update = picard_jtx_update_fn(request)
    if memory == 0:
        return update
    ops = request.backend
    rows = len(request.nodes)
    for name, values in (
        ("base proposal means", update.base_proposal_means),
        ("exact_target_means", update.exact_target_means),
    ):
        _check_stack(name, values, request.parent_states, rows, ops)
    errors = update.exact_target_means - update.base_proposal_means
    factors_by_row = []
    for i, (image, node, step) in enumerate(zip(
        request.indices_in_batch, request.nodes, request.steps
    )):
        key = ("broyden", id(request.target), id(request.proposal), image, node, step)
        previous = request.history.get(key)
        state, error = request.parent_states[i], errors[i]
        factors = ()
        if (request.iteration > 0 and previous is not None
                and previous.iteration == request.iteration - 1):
            factors = previous.factors[-memory:]
            displacement = state - previous.state
            length = ops.norm(displacement)
            scale = max(1.0, ops.norm(state), ops.norm(previous.state))
            threshold = math.sqrt(ops.finfo_eps(state)) * scale
            if math.isfinite(length) and length > threshold:
                # Reserve space first so the retained B satisfies the newest secant.
                retained = factors[-(memory - 1):] if memory > 1 else ()
                direction = displacement / length
                u = (error - previous.error) / length - _apply_broyden(
                    retained, direction, ops
                )
                if ops.is_finite(u) and ops.is_finite(direction):
                    factors = retained + ((u, direction),)
        request.history[key] = _BroydenHistory(
            request.iteration, ops.copy(state), ops.copy(error), factors
        )
        factors_by_row.append(factors)
    return RefinementUpdate(
        base_proposal_means=update.base_proposal_means,
        exact_target_means=update.exact_target_means,
        broyden_factors=tuple(factors_by_row),
    )


def _check_stack(name: str, value: Array, reference: Array, rows: int, ops: Backend) -> None:
    expected = tuple(reference.shape)
    actual = tuple(getattr(value, "shape", ()))
    if actual != expected or not actual or int(actual[0]) != rows:
        raise ValueError(f"{name} has shape {actual}; expected {expected}")
    ops.check_state_dtype(value, name)
    if not ops.is_finite(value):
        raise ValueError(f"{name} contains non-finite values")


def _rebuild_from_drifts(
    update: RefinementUpdate,
    target: TargetTransition,
    layout: RefinementLayout,
    level: RefinementLevel,
    anchors: Array,
    parent_states: Array,
    ops: Backend,
) -> Array:
    """Proposal means of one level's rebuilt parents from frozen drifts.

    The mean is ``apply(D, X^(j+1))``. With exact target means it is computed
    as ``m(X^(j)) + [apply(D, X^(j+1)) - apply(D, X^(j))]``: the same value in
    exact arithmetic, but bitwise the cached target mean wherever the parent
    has not moved, so converged parents verify with ``delta = 0`` exactly.
    """

    drifts = ops.take(update.drifts, level.parent_positions)
    steps = tuple(layout.steps[p] for p in level.parent_positions)

    def apply(states: Array) -> Array:
        means = target.apply_drift(drifts, states, steps)
        _check_stack("apply_drift means", means, anchors, len(level.parent_ids), ops)
        return means

    means = apply(anchors)
    if update.exact_target_means is None:
        return means
    snapshot = ops.take(parent_states, level.parent_positions)
    return ops.take(update.exact_target_means, level.parent_positions) + (
        means - apply(snapshot)
    )


def refine_tree(
    *,
    states: Array,
    proposal_means: Array,
    scaled_innovations: Array,
    layout: RefinementLayout,
    iterations: int,
    update_fn: RefinementUpdateFn,
    target: TargetTransition,
    ops: Backend,
    proposal: Optional[ProposalTransition] = None,
) -> Optional[ExactTargetCache]:
    """Run ``iterations`` synchronous sweeps and return the last target cache."""

    if iterations == 0 or not layout.internal_ids:
        return None

    last_cache: Optional[ExactTargetCache] = None
    history = {}

    for iteration in range(iterations):
        # ``take`` is a copy by Backend contract, so reconstruction below cannot
        # mutate the frozen X^(j) snapshot seen by the callback or cache.
        parent_states = ops.take(states, layout.internal_ids)
        request = RefinementRequest(
            iteration=iteration,
            indices_in_batch=layout.indices_in_batch,
            nodes=layout.logical_nodes,
            steps=layout.steps,
            parent_states=parent_states,
            current_proposal_means=ops.take(proposal_means, layout.internal_ids),
            sigmas=layout.sigmas,
            target=target,
            backend=ops,
            proposal=proposal,
            history=history,
        )
        update = update_fn(request)
        if not isinstance(update, RefinementUpdate):
            raise TypeError("refinement_update_fn must return RefinementUpdate")
        if update.base_proposal_means is not None:
            if proposal is None:
                raise ValueError("base_proposal_means requires the base proposal")
            name, values = "base proposal means", update.base_proposal_means
        elif update.drifts is not None:
            name, values = "refinement drifts", update.drifts
        else:
            name, values = "refinement increments", update.increments
        _check_stack(name, values, parent_states, len(layout.internal_ids), ops)
        if update.broyden_factors is not None:
            if len(update.broyden_factors) != len(layout.internal_ids):
                raise ValueError("broyden_factors needs one entry per internal node")
            for row, factors in enumerate(update.broyden_factors):
                for u, v in factors:
                    for factor in (u, v):
                        _check_stack(
                            "Broyden factor", factor[None], parent_states[row:row + 1],
                            1, ops,
                        )

        if update.exact_target_means is None:
            last_cache = None
        else:
            _check_stack(
                "exact_target_means", update.exact_target_means, parent_states,
                len(layout.internal_ids), ops,
            )
            last_cache = ExactTargetCache(
                target=target,
                flat_ids=layout.internal_ids,
                indices_in_batch=layout.indices_in_batch,
                nodes=layout.logical_nodes,
                steps=layout.steps,
                states=ops.copy(parent_states),
                means=ops.copy(update.exact_target_means),
            )

        # Roots stay fixed. Because levels are breadth-first, every new parent
        # has already been reconstructed when its children are visited.
        for level in layout.levels:
            anchors = ops.take(states, level.parent_ids)
            if update.base_proposal_means is not None:
                positions = level.parent_positions
                base_means = proposal.means(
                    tuple(layout.indices_in_batch[p] for p in positions),
                    anchors,
                    tuple(layout.steps[p] for p in positions),
                )
                _check_stack(
                    "rebuilt base proposal means", base_means, anchors,
                    len(level.parent_ids), ops,
                )
                means = ops.take(update.exact_target_means, positions) + (
                    base_means - ops.take(update.base_proposal_means, positions)
                )
            elif update.drifts is None:
                means = anchors + ops.take(update.increments, level.parent_positions)
            else:
                means = _rebuild_from_drifts(
                    update, target, layout, level, anchors, parent_states, ops
                )
            if update.broyden_factors is not None:
                for row, position in enumerate(level.parent_positions):
                    factors = update.broyden_factors[position]
                    if not factors:
                        continue
                    displacement = anchors[row] - parent_states[position]
                    corrected = means[row] + _apply_broyden(factors, displacement, ops)
                    # Row-local fallback preserves the same conditional Gaussian law.
                    if ops.is_finite(corrected):
                        ops.put(means, [row], corrected[None])
            ops.put(proposal_means, level.parent_ids, means)
            children = (
                ops.repeat_rows(means, level.child_counts)
                + ops.take(scaled_innovations, level.child_innovation_ids)
            )
            ops.put(states, level.child_ids, children)

    return last_cache


def reusable_target_rows(
    cache: Optional[ExactTargetCache],
    *,
    target: TargetTransition,
    final_states: Array,
    ops: Backend,
) -> dict:
    """Return cache rows whose full identity and represented state still match."""

    if cache is None or cache.target is not target:
        return {}
    current = ops.take(final_states, cache.flat_ids)
    equal = ops.equal_rows(cache.states, current)
    out = {}
    for i, same in enumerate(equal):
        if same:
            key = (
                cache.flat_ids[i], cache.indices_in_batch[i],
                cache.nodes[i], cache.steps[i],
            )
            out[key] = cache.means[i]
    return out


def reusable_exact_target_mean(
    cache: Optional[ExactTargetMean],
    *,
    target: TargetTransition,
    index_in_batch: int,
    step: int,
    state: Array,
    ops: Backend,
) -> Optional[Array]:
    """Return a cross-round target value only when its full identity matches."""

    if (
        cache is None
        or cache.target is not target
        or cache.index_in_batch != index_in_batch
        or cache.step != step
    ):
        return None
    same = ops.equal_rows(
        ops.stack_rows([cache.state]), ops.stack_rows([state])
    )
    return cache.mean if same[0] else None
