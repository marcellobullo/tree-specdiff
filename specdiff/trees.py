"""Draft topologies.

Appendix C makes the key observation that lets one class cover every case: the
reverse chain is Markov, so each drafted state is drawn conditionally on
exactly one parent, and therefore *any* draft set is a rooted tree. RMC's chain
is the tree with ``K = 1``; a single-proposal list is ``L = 1``; the paper's
uniform family is ``|C(u)| = K`` for every node of depth ``< L``. Pruned or
depth-dependent trees need no special handling.

Nodes are integers in breadth-first order, so node ``0`` is the root and the
set of nodes of depth ``<= m`` is always a prefix ``0..offset[m + 1]``. That
makes the truncation ``T|_m`` of eq. (27), needed in the last few rounds when
``L_n = min(L, N - n) < L``, a slice rather than a rebuild.
"""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

ROOT = 0


class DraftTree:
    """A finite rooted tree ``T = (V, pa)`` describing one round's draft set.

    Parameters
    ----------
    parents:
        ``parents[0]`` must be ``-1`` (the root) and ``parents[u] < u`` for
        ``u > 0``. The second condition both rules out cycles and forces the
        breadth-first ordering the class relies on.
    """

    __slots__ = ("_parents", "_depth", "_children", "_layers", "_internal", "_max_depth", "_trunc_cache")

    def __init__(self, parents: Sequence[int]) -> None:
        parents = tuple(int(p) for p in parents)
        if not parents or parents[0] != -1:
            raise ValueError("parents[0] must be -1 (the root)")
        depth = [0] * len(parents)
        children: list[list[int]] = [[] for _ in parents]
        for u in range(1, len(parents)):
            p = parents[u]
            if not (0 <= p < u):
                raise ValueError(f"node {u}: parent {p} must satisfy 0 <= parent < node")
            depth[u] = depth[p] + 1
            children[p].append(u)
        if any(depth[u] < depth[u - 1] for u in range(1, len(parents))):
            raise ValueError("nodes must be listed in breadth-first (non-decreasing depth) order")

        self._parents = parents
        self._depth = tuple(depth)
        self._children = tuple(tuple(c) for c in children)
        self._max_depth = max(depth)
        layers: list[list[int]] = [[] for _ in range(self._max_depth + 1)]
        for u, d in enumerate(depth):
            layers[d].append(u)
        self._layers = tuple(tuple(l) for l in layers)
        self._internal = tuple(u for u, c in enumerate(self._children) if c)
        self._trunc_cache = {}

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_widths(cls, widths: Sequence[int]) -> "DraftTree":
        """Tree where every node at depth ``l`` has ``widths[l]`` children.

        ``from_widths([3, 2])`` is a tree with 3 children at the root and 2
        children under each of those: ``L = 2``, ``B = 9``.
        """
        widths = [int(w) for w in widths]
        if any(w < 1 for w in widths):
            raise ValueError("widths must be >= 1")
        parents = [-1]
        frontier = [ROOT]
        for w in widths:
            new_frontier = []
            for u in frontier:
                for _ in range(w):
                    parents.append(u)
                    new_frontier.append(len(parents) - 1)
            frontier = new_frontier
        return cls(parents)

    @classmethod
    def uniform(cls, branching: int, lookahead: int) -> "DraftTree":
        """The paper's ``(K, L)`` family: ``B = K + ... + K^L`` (eq. 12)."""
        if branching < 1 or lookahead < 1:
            raise ValueError("branching and lookahead must be >= 1")
        return cls.from_widths([branching] * lookahead)

    @classmethod
    def chain(cls, lookahead: int) -> "DraftTree":
        """RMC's topology: ``K = 1``, a single linear lookahead."""
        return cls.uniform(1, lookahead)

    @classmethod
    def largest_uniform(cls, budget: int, branching: int) -> "DraftTree":
        """Deepest ``(K, L)`` tree fitting a proposal budget, per eq. (12)."""
        if budget < branching:
            raise ValueError(f"budget {budget} cannot fit even one level of width {branching}")
        total, lookahead = 0, 0
        while True:
            level_size = branching**(lookahead + 1)
            if total + level_size > budget:
                break
            total += level_size
            lookahead += 1
        return cls.uniform(branching, lookahead)

    # -------------------------------------------------------------- inspection
    @property
    def size(self) -> int:
        """``|V|``, including the root."""
        return len(self._parents)

    @property
    def budget(self) -> int:
        """``B = |V| - 1``: the states this round **drafts**.

        The paper's proposal budget (eq. 12, ``B = K + ... + K^L`` when
        uniform), and the x-axis every speedup is plotted against. It is a
        *proposal* cost: under the self-speculative delayed drift, drafting a
        state is a vector add, not a network call.

        The expensive budget is :meth:`verification_budget`, which counts the
        states the **target** sees. The two differ by a factor of ``K``, so do
        not read ``budget`` as the hardware requirement.
        """
        return self.size - 1

    def verification_budget(self, *, evaluate_leaves: bool = False) -> int:
        """States the **target** must evaluate per round.

        This is the expensive budget, and the one that sets the batch a round
        has to fit in memory. Contrast :attr:`budget`, which counts drafted
        states.

        ``evaluate_leaves=False`` (default)
            ``|I|``, the internal nodes -- what the sampler actually evaluates.
            Leaves are never parents, so their target means are never needed to
            verify anything (eq. 26: ``|I| = B / K`` when uniform).
        ``evaluate_leaves=True``
            ``|I|`` plus the leaf level. A delayed-drift proposal can use the
            leaf drifts to carry an *exact* drift into the next round on full
            acceptance, worth a few percent of NFE speedup; it costs ``K^L``
            extra rows, taking the batch from ``B / K`` to ``B + 1``.

        The ``+1`` is the root. specdiff evaluates it, because its target mean
        is what verifies the depth-1 children. An implementation whose carry is
        exact *at* the root can skip it and land on exactly ``B``.
        """
        return self.size if evaluate_leaves else len(self._internal)

    @property
    def depth(self) -> int:
        """``L``, the lookahead."""
        return self._max_depth

    @property
    def internal_nodes(self) -> Tuple[int, ...]:
        """``I(T)``: nodes with children. These, and only these, are the states
        the target model is evaluated at (eq. 26: ``|I| = B / K`` when uniform).

        See :meth:`verification_budget` for the count, and for what changes if
        the leaf level is evaluated too."""
        return self._internal

    def parent(self, u: int) -> int:
        return self._parents[u]

    def children(self, u: int) -> Tuple[int, ...]:
        """``C(u)``, in a fixed order that is also the drafting order."""
        return self._children[u]

    def depth_of(self, u: int) -> int:
        return self._depth[u]

    def layer(self, level: int) -> Tuple[int, ...]:
        """``V_l``, the nodes of depth ``level``."""
        if level < 0 or level > self._max_depth:
            return ()
        return self._layers[level]

    def width_at(self, level: int) -> int:
        """Number of children of a node at depth ``level``.

        Defined only for level-uniform trees, which is the condition batched
        verification needs: at a given level every trajectory in the batch must
        present the same number of candidates, or the request cannot be a
        rectangular ``(batch, K, *state_shape)`` array.
        """
        nodes = self.layer(level)
        if not nodes:
            return 0
        widths = {len(self._children[u]) for u in nodes}
        if len(widths) > 1:
            raise ValueError(
                f"level {level} has nodes of differing width {sorted(widths)}; "
                "width_at is only defined for level-uniform trees"
            )
        return widths.pop()

    def is_level_uniform(self) -> bool:
        """True if every node at a given depth has the same number of children.

        Weaker than :meth:`is_uniform`: ``from_widths([3, 1, 2])`` is
        level-uniform but not uniform.
        """
        for level in range(self.depth + 1):
            if len({len(self._children[u]) for u in self.layer(level)}) > 1:
                return False
        return True

    def is_uniform(self) -> bool:
        widths = {len(self._children[u]) for u in self._internal}
        return len(widths) <= 1

    @property
    def branching(self) -> int:
        """``K`` for a uniform tree; the maximum width otherwise."""
        return max((len(c) for c in self._children), default=0)

    # ------------------------------------------------------------- truncation
    def truncate(self, max_depth: int) -> "DraftTree":
        """``T|_m`` of eq. (27): the subtree induced by nodes of depth ``<= m``.

        Used for the final rounds, where ``L_n = min(L, N - n) < L``. Cached
        per instance, since a run touches at most ``L`` distinct truncations.
        """
        if max_depth >= self._max_depth:
            return self
        if max_depth < 1:
            raise ValueError("a draft tree must keep at least one level")
        cached = self._trunc_cache.get(max_depth)
        if cached is None:
            keep = sum(len(self._layers[d]) for d in range(max_depth + 1))
            cached = DraftTree(self._parents[:keep])
            self._trunc_cache[max_depth] = cached
        return cached

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.size))

    def __repr__(self) -> str:
        shape = "uniform" if self.is_uniform() else "irregular"
        return (
            f"DraftTree({shape}, K={self.branching}, L={self.depth}, "
            f"B={self.budget}, |I|={len(self._internal)})"
        )
