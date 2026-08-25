"""Gate for `lazy.py`: does the simulator agree with the eager sampler?

The two consume different numbers of normal draws, so a given seed gives
different trajectories and the comparison has to be statistical. Run enough
replicates that the standard error on mean `target_calls` is small relative to
the difference you would care about (~0.1 calls), and check every combination
of `prefetch` and `evaluate_leaves` -- the prefetch policy is duplicated
between the two implementations, which is exactly the kind of thing that drifts.

    python experiments/gm/validate_lazy.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# parents[2] is the repo root: `import experiments.gm.models` needs it on
# the path, since neither `experiments` nor `experiments/gm` is a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lazy  # noqa: E402
import experiments.gm.models as models  # noqa: E402

from specdiff import (  # noqa: E402
    DelayedDriftProposal,
    DraftTree,
    SpeculativeSampler,
    create_verifier,
)

SEED = 20260714


def eager(setting, rule, K, L, n, prefetch, evaluate_leaves):
    sampler = SpeculativeSampler(
        target=setting.target, proposal=DelayedDriftProposal(setting.target),
        schedule=setting.schedule, tree=DraftTree.uniform(K, L),
        verifier=create_verifier(rule), num_steps=setting.num_steps,
        prefetch=prefetch, evaluate_leaves=evaluate_leaves,
    )
    calls, rows = [], []
    for t in range(n):
        i, r = np.random.SeedSequence(SEED, spawn_key=(K, L, t)).spawn(2)
        res = sampler.sample(setting.initial_state(np.random.default_rng(i)),
                             rng=np.random.default_rng(r))
        calls.append(res.target_calls)
        rows.append(res.target_states_evaluated)
    return np.array(calls, float), np.array(rows, float)


def lazily(setting, rule, K, L, n, prefetch, evaluate_leaves):
    calls, rows = [], []
    for t in range(n):
        i, r = np.random.SeedSequence(SEED, spawn_key=(K, L, t)).spawn(2)
        res = lazy.simulate(setting, rule, K, L,
                            setting.initial_state(np.random.default_rng(i)),
                            np.random.default_rng(r), prefetch=prefetch,
                            evaluate_leaves=evaluate_leaves)
        calls.append(res.target_calls)
        rows.append(res.target_rows)
    return np.array(calls, float), np.array(rows, float)


def main() -> None:
    setting = models.build()
    reps = 400
    cases = [(rule, K, L) for rule in ("d-grs",) for K, L in ((1, 3), (2, 3), (4, 3), (3, 4))]
    cases += [("rmc", 1, 3), ("rmc", 1, 8)]

    print(f"{reps} replicates per cell.  z on mean target_calls; |z| > 4 fails.\n")
    print(f"{'rule':>6}{'K':>3}{'L':>3}{'prefetch':>10}{'leaves':>8} | "
          f"{'eager':>8}{'lazy':>8}{'z':>7} | {'rows e':>8}{'rows l':>8}")
    worst, bad = 0.0, 0
    for rule, K, L in cases:
        for prefetch in ("none", "parent", "nearest"):
            for leaves in (False, True):
                ce, re_ = eager(setting, rule, K, L, reps, prefetch, leaves)
                cl, rl = lazily(setting, rule, K, L, reps, prefetch, leaves)
                se = math.sqrt(ce.var(ddof=1) / reps + cl.var(ddof=1) / reps)
                z = (cl.mean() - ce.mean()) / se if se > 0 else 0.0
                worst = max(worst, abs(z))
                flag = "" if abs(z) <= 4 else "   <-- FAIL"
                bad += abs(z) > 4
                print(f"{rule:>6}{K:>3}{L:>3}{prefetch:>10}{str(leaves):>8} | "
                      f"{ce.mean():>8.2f}{cl.mean():>8.2f}{z:>7.2f} | "
                      f"{re_.mean():>8.1f}{rl.mean():>8.1f}{flag}")
    print(f"\nworst |z| = {worst:.2f} over {len(cases) * 6} configurations; {bad} failing")


if __name__ == "__main__":
    main()
