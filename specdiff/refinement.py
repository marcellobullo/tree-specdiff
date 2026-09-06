"""Synchronous Picard refinement for drafted diffusion trees.

The sampler owns topology and buffer layout. This module evaluates one
row-local update for every internal node from a frozen snapshot, then
reconstructs all children breadth-first using fixed edge innovations.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Optional, Tuple

from .kernels import TargetTransition
from .ops import Backend

Array = Any


@dataclass(frozen=True)
class RefinementRequest:
    """Inputs to one synchronous refinement sweep.

    Output row ``i`` must depend only on input row ``i``, fixed model
    parameters, and conditioning for ``indices_in_batch[i]``. Mixing rows,
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


@dataclass(frozen=True)
class RefinementUpdate:
    """A row-local increment and optional correctness-bearing target cache.

    If supplied, ``exact_target_means[i]`` asserts that it is exactly the
    target mean at ``request.parent_states[i]`` for the associated image and
    step. A false assertion can invalidate exact sampling in the same way as
    an incorrect custom verifier. ``check_contract=True`` validates reused
    cache rows.
    """

    increments: Array
    exact_target_means: Optional[Array] = None


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
    """Canonical target-backed Picard increment ``m^q_s(x) - x``."""

    means = request.target(
        request.indices_in_batch,
        request.parent_states,
        request.steps,
    )
    return RefinementUpdate(
        increments=means - request.parent_states,
        exact_target_means=means,
    )


def _check_stack(name: str, value: Array, reference: Array, rows: int, ops: Backend) -> None:
    expected = tuple(reference.shape)
    actual = tuple(getattr(value, "shape", ()))
    if actual != expected or not actual or int(actual[0]) != rows:
        raise ValueError(f"{name} has shape {actual}; expected {expected}")
    ops.check_state_dtype(value, name)
    if not ops.is_finite(value):
        raise ValueError(f"{name} contains non-finite values")


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
) -> Optional[ExactTargetCache]:
    """Run ``iterations`` synchronous sweeps and return the last target cache."""

    if iterations == 0 or not layout.internal_ids:
        return None

    last_cache: Optional[ExactTargetCache] = None

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
        )
        update = update_fn(request)
        if not isinstance(update, RefinementUpdate):
            raise TypeError("refinement_update_fn must return RefinementUpdate")
        _check_stack(
            "refinement increments", update.increments, parent_states,
            len(layout.internal_ids), ops,
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
            means = (
                ops.take(states, level.parent_ids)
                + ops.take(update.increments, level.parent_positions)
            )
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
