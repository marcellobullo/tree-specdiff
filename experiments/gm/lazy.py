"""Lazy round simulation: realise only the committed branch of the draft tree.

Purpose
-------
:class:`~specdiff.sampler.SpeculativeSampler` drafts every node of the tree,
because it is a *sampler*: it must produce the trajectory, and Algorithm 3's
single batched target call covers all internal nodes at once. Consequently,
``(7, 7)`` needs approximately 7.9 GB for 960,800
states and as many proposal means.

Only the committed branch survives a round. The trajectory depends
only on the committed node at each level and that node's ``K`` children, which is
``K * L = 49`` states at ``(7, 7)``. This simulator uses that property to make
the largest sweep configurations tractable.

Limitations
-----------
This is a cost simulator, not a sampler. Realising the branch lazily means
the target mean of each level's parent is needed *sequentially* -- you cannot
know which parent to evaluate until the level above has been verified -- so a
faithful lazy sampler would spend ``L`` calls per round instead of one, which is
the behavior Algorithm 3 avoids. This module evaluates on demand and counts one
call per round, which is valid for NFE accounting but not wall-clock estimates.

Two consequences follow:

* ``target_rows`` is **analytic** here -- the count the eager sampler *would*
  have made (``verification_budget`` of the truncated tree per round) -- not a
  measurement, since the rows are never materialised.
* The trajectory law matches the eager sampler but individual trajectories do
  not: far fewer normal draws are consumed, so the same seed gives a different
  path. Validation against the eager sampler is therefore statistical; see
  ``validate_lazy.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# parents[2] is the repo root: `import experiments.gm.models` needs it on
# the path, since neither `experiments` nor `experiments/gm` is a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from specdiff import VerifyRequest, create_verifier  # noqa: E402
from specdiff.ops import resolve_backend  # noqa: E402


def verification_budget(K: int, depth: int, evaluate_leaves: bool = False) -> int:
    """``DraftTree.uniform(K, depth).verification_budget(...)``, in closed form.

    Constructing the tree only to count nodes would allocate approximately
    420 MB per worker at ``(7, 7)``. The closed form is
    ``|I| = 1 + K + ... + K^(depth-1)``, plus ``K^depth``
    when the leaf level is evaluated.
    """
    internal = depth if K == 1 else (K**depth - 1) // (K - 1)
    return internal + (K**depth if evaluate_leaves else 0)


@dataclass(frozen=True)
class LazyResult:
    target_calls: int
    target_rows: int
    rounds: int
    accepted_depth: int
    sample: np.ndarray


def simulate(setting, rule, K, L, init, rng, *, prefetch="nearest",
             evaluate_leaves=False) -> LazyResult:
    """One trajectory, materialising only the committed branch.

    Mirrors ``SpeculativeSampler._round`` phase for phase; the only differences
    are which states get drafted and how the cost is counted.
    """
    ops = resolve_backend(init)
    target, schedule, N = setting.target, setting.schedule, setting.num_steps
    verifier = create_verifier(rule)
    if verifier.max_children is not None and K > verifier.max_children:
        raise ValueError(
            f"{rule} supports at most K={verifier.max_children} proposals per node, "
            f"but K={K}. Use K=1 for single-proposal rules."
        )
    verifier.reset()

    state = init
    n = calls = rows = rounds = accepted_total = 0
    drift: Optional[np.ndarray] = None       # what DelayedDriftProposal holds
    known_root_mean: Optional[np.ndarray] = None

    while n < N:
        lookahead = min(L, N - n)

        # --- the proposal's frozen drift (DelayedDriftProposal.on_round_start)
        if drift is None or prefetch == "none":
            root_mean = target.means((0,), state[None], (n,))
            drift = target.freeze_drift(state[None], root_mean, (n,))[0]
            calls += 1
            rows += 1
            known_root_mean = None

        # --- Phase 2, as one call. The rows are what the eager sampler would
        # --- have evaluated; here they are computed on demand below.
        calls += 1
        rows += verification_budget(K, lookahead, evaluate_leaves)
        if known_root_mean is not None:
            rows -= 1  # the exact-root optimisation

        # --- Phase 3: descend, drafting one level at a time
        parent, parent_mean = state, known_root_mean
        known_root_mean = None
        committed = accepted = 0
        rejected = False
        # The node whose children were verified last -- NOT the node we descended
        # into. On full acceptance those differ by one level, and conflating them
        # would silently turn `prefetch="parent"` into the exact-leaf case.
        verified_parent = verified_mean = verified_children = None

        for level in range(1, lookahead + 1):
            step = n + level - 1
            sigma = schedule(step)
            proposal_mean = target.apply_drift(drift[None], parent[None], (step,))[0]
            if parent_mean is None:
                parent_mean = target.means((0,), parent[None], (step,))[0]

            children = proposal_mean + sigma * ops.randn_stack(K, init, rng)
            result = verifier(VerifyRequest(
                step=step, proposal_mean=proposal_mean, target_mean=parent_mean,
                sigma=sigma, children=children, parent_state=parent, rng=rng,
                info={"level": level},
            ))
            verified_parent, verified_mean, verified_children = parent, parent_mean, children
            state = result.state
            committed += 1
            if not result.accepted:
                rejected = True
                break
            accepted += 1
            parent, parent_mean = state, None

        # --- the prefetch policy, mirroring SpeculativeSampler._prefetch_nearest
        if prefetch == "parent":
            drift = target.freeze_drift(
                verified_parent[None], verified_mean[None], (n + committed - 1,)
            )[0]
        elif prefetch == "nearest":
            drift, known_root_mean = _prefetch_nearest(
                ops, target, state, verified_parent, verified_mean, verified_children,
                n, committed, lookahead, rejected, evaluate_leaves,
            )

        n += committed
        rounds += 1
        accepted_total += accepted

    return LazyResult(calls, rows, rounds, accepted_total, state)


def _prefetch_nearest(ops, target, committed_state, parent, parent_mean, children,
                      n, committed, lookahead, rejected, evaluate_leaves):
    """Returns ``(drift, known_root_mean)``; see the sampler's version."""
    depth = committed  # the committed state sits at this depth
    freeze = lambda x, m, step: target.freeze_drift(x[None], m[None], (step,))[0]  # noqa: E731

    if not rejected and evaluate_leaves:
        # Case 1: the committed leaf's own drift -- exact at the next root, and
        # therefore also the next root's target mean.
        mean = target.means((0,), committed_state[None], (n + depth,))[0]
        return freeze(committed_state, mean, n + depth), mean

    if rejected and (depth <= lookahead - 1 or evaluate_leaves):
        # Case 2: nearest drafted sibling at the committed depth. Its siblings
        # are internal (so already paid for) unless the rejection was at the
        # last level, which only `evaluate_leaves` covers.
        means = target.means((0,) * len(children), children, (n + depth,) * len(children))
        d = [ops.norm(children[j] - committed_state) for j in range(len(children))]
        j = int(np.argmin(d))
        return freeze(children[j], means[j], n + depth), None

    # Case 3: the last verified parent's, one step stale.
    return freeze(parent, parent_mean, n + depth - 1), None
