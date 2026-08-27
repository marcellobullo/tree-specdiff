"""Run the Section 5.1 ``(K, L)`` sweep for both figures.

Runs every ``(K, L)`` cell for both of the paper's rules and writes one row per
trajectory to a resumable CSV. Plotting is handled separately by ``plot_gm.py``.

    python experiments/gm/gm_sweep.py --out results/gm/$(date +%Y%m%d-%H%M%S)

Matched budgets
---------------
Each D-GRS cell runs ``DraftTree.uniform(K, L)``; its RMC counterpart runs a
chain matched under ``--match``, which defaults to ``verification``: equal
*target* batch, ``chain(|I|)``. That is the hardware-matched arm, since ``|I|``
is what has to fit in memory and what the expensive network actually evaluates.
``--match budget`` gives the paper's protocol instead, equal *proposal* budget
``B = K + ... + K^L`` (eq. 12); it is what produced the committed
``results/figure3``, and at ``d=512, eps=0.06`` the two differ by 1.27x vs
1.77x, so the choice materially affects the comparison. A chain deeper than the horizon is unnecessary:
a round starting at step ``n`` truncates to ``min(depth, N - n)`` -- so the
chain is built at ``min(B, N)``, identical in behaviour and vastly cheaper to
construct than ``chain(960799)``.

Leaves
------
Only internal nodes are verified (eq. 26: ``|I| = B / K``), so the leaf level is
not evaluated. Including leaves can improve NFE speedup by a few percent at a
factor-of-``K`` increase in target rows. ``--prefetch`` selects
what the delayed-drift proposal reuses between rounds; ``nearest`` is the
default everywhere and needs no extra evaluation.

Two budgets
-----------
``B = tree.budget`` is the **proposal** budget of eq. (12) -- the states a round
drafts, and the x-axis of the figure. Drafting is a vector add under the delayed
drift, so this is not what the hardware has to fit.

``tree.verification_budget()`` is the **target** budget: ``|I| = B / K`` here,
because the leaf level is not evaluated. That is the batch a round must hold,
and it is the number to read when asking whether a cell is runnable.

Cost metrics
------------
Both are recorded, because they answer different questions:

``target_calls``
    NFEs -- batched calls, the paper's metric. Assumes the batch is free, which
    holds when the run is latency-bound with room for ``|I|`` rows.
``target_rows``
    States actually pushed through the target over the whole trajectory. The
    RMC chain saturates near the horizon while the D-GRS tree keeps growing
    with ``K``. This difference is part of the comparison: a chain
    cannot spend a budget deeper than the trajectory is long, so extra hardware
    buys it nothing. Absorbing that budget through width is what the tree adds.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
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

FIELDS = ("rule", "K", "L", "B", "verification_budget", "chain_depth", "replicate",
          "target_calls", "target_rows", "rounds", "accepted_depth",
          "num_steps", "total_steps")


def trajectory_rngs(seed: int, K: int, L: int, replicate: int):
    """Return independent, indexable streams shared by both rules.

    Keyed by ``(K, L, replicate)`` rather than spawned in sequence, so re-running
    one cell after an interruption reproduces it exactly without knowing how
    many cells ran before. Sharing the key across rules pairs the comparison on
    the initial state; varying it across cells keeps each cell an independent
    estimate, so a flat panel shows its own Monte-Carlo floor.
    """
    init_ss, run_ss = np.random.SeedSequence(seed, spawn_key=(K, L, replicate)).spawn(2)
    return np.random.default_rng(init_ss), np.random.default_rng(run_ss)


def matched_chain_depth(tree, num_steps, match, evaluate_leaves):
    """Depth of the RMC chain that matches `tree` under the chosen protocol.

    ``match="budget"``
        Equal **proposal** budget: a chain of `B` drafted states. This is the
        paper's protocol.
    ``match="verification"``
        Equal **verification** budget: a chain whose target batch is the same
        size as the tree's. Since a chain of depth `m` evaluates `m` nodes
        (`m + 1` with leaves), that is `|I|` and `B` respectively.

    Either way the chain is clamped to the horizon: a round truncates to
    ``min(depth, N - n)``, so depth beyond `N` is unreachable and building it
    would only cost memory. That clamp is also why the two protocols mostly
    coincide -- once the matched depth exceeds `N`, both give `chain(N)`.
    """
    if match == "budget":
        depth = tree.budget
    else:
        depth = tree.verification_budget(evaluate_leaves=evaluate_leaves)
        if evaluate_leaves:
            depth -= 1  # a chain of depth m evaluates m + 1 nodes with leaves
    return max(1, min(depth, num_steps))


def _matched_depth(K, L, budget, num_steps, match, evaluate_leaves):
    """`matched_chain_depth` without constructing the tree."""
    if match == "budget":
        depth = sum(K**i for i in range(1, L + 1))
    else:
        depth = lazy.verification_budget(K, L, evaluate_leaves)
        if evaluate_leaves:
            depth -= 1
    return max(1, min(depth, num_steps))


def build_sampler(setting, rule, K, L, prefetch, evaluate_leaves, match):
    """The eager sampler for one cell, plus the chain depth it was matched at."""
    uniform = DraftTree.uniform(K, L)
    depth = matched_chain_depth(uniform, setting.num_steps, match, evaluate_leaves)
    tree = uniform if rule == "d-grs" else DraftTree.chain(depth)
    sampler = SpeculativeSampler(
        target=setting.target,
        proposal=DelayedDriftProposal(setting.target),
        schedule=setting.schedule,
        tree=tree,
        verifier=create_verifier(rule),
        num_steps=setting.num_steps,
        prefetch=prefetch,
        evaluate_leaves=evaluate_leaves,
    )
    return sampler, tree, depth


def one_trajectory(setting, rule, K, L, replicate, cfg, sampler=None, tree=None):
    """One trajectory, eager or lazy. Returns the CSV row.

    Deterministic in ``(K, L, replicate)`` alone -- not in evaluation order --
    which is what lets this run under any number of workers and still produce
    byte-identical results.
    """
    init_rng, run_rng = trajectory_rngs(cfg["seed"], K, L, replicate)
    budget = sum(K**i for i in range(1, L + 1))
    prefetch, leaves, match = cfg["prefetch"], cfg["evaluate_leaves"], cfg["match"]

    if cfg["lazy"]:
        depth = _matched_depth(K, L, budget, setting.num_steps, match, leaves)
        sim_K, sim_L = (K, L) if rule == "d-grs" else (1, depth)
        res = lazy.simulate(setting, rule, sim_K, sim_L,
                            setting.initial_state(init_rng), run_rng,
                            prefetch=prefetch, evaluate_leaves=leaves)
        calls, rows = res.target_calls, res.target_rows
        rounds, accepted = res.rounds, res.accepted_depth
        verif = lazy.verification_budget(sim_K, sim_L, leaves)
    else:
        depth = matched_chain_depth(DraftTree.uniform(K, L), setting.num_steps,
                                    match, leaves)
        res = sampler.sample(setting.initial_state(init_rng), rng=run_rng)
        calls, rows = res.target_calls, res.target_states_evaluated
        rounds = len(res.rounds)
        accepted = sum(r.accepted_depth for r in res.rounds)
        verif = tree.verification_budget(evaluate_leaves=leaves)

    return {
        "rule": rule, "K": K, "L": L, "B": budget,
        "verification_budget": verif,
        "chain_depth": depth if rule != "d-grs" else "",
        "replicate": replicate, "target_calls": calls, "target_rows": rows,
        "rounds": rounds, "accepted_depth": accepted,
        "num_steps": setting.num_steps, "total_steps": setting.total_steps,
    }


# --------------------------------------------------------------- worker pool
_WORKER: dict = {}


def _init_worker(cfg):
    """Build the model once per worker; samplers are cached per cell below."""
    _WORKER["cfg"] = cfg
    _WORKER["setting"] = models.build(
        cfg["dimension"], cfg["num_components"], cfg["num_steps"], cfg["eps"],
        mixture_seed=cfg["mixture_seed"])
    _WORKER["samplers"] = {}


def _job(args):
    rule, K, L, replicate = args
    cfg, setting = _WORKER["cfg"], _WORKER["setting"]
    sampler = tree = None
    if not cfg["lazy"]:
        key = (rule, K, L)
        if key not in _WORKER["samplers"]:
            _WORKER["samplers"].clear()  # one cell at a time; trees are large
            sampler, tree, _ = build_sampler(setting, rule, K, L, cfg["prefetch"],
                                             cfg["evaluate_leaves"], cfg["match"])
            _WORKER["samplers"][key] = (sampler, tree)
        sampler, tree = _WORKER["samplers"][key]
    return one_trajectory(setting, rule, K, L, replicate, cfg, sampler, tree)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/figure3")
    p.add_argument("--dimension", type=int, default=512)
    p.add_argument("--num-components", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=30, help="T, before dropping endpoints")
    p.add_argument("--eps", type=float, default=0.06)
    p.add_argument("--mixture-seed", type=int, default=20260714)
    p.add_argument("--seed", type=int, default=20260714, help="sampling seed")
    p.add_argument("--K-values", type=int, nargs="+", default=list(range(1, 8)))
    p.add_argument("--L-values", type=int, nargs="+", default=list(range(1, 8)))
    p.add_argument("--rules", nargs="+", default=["d-grs", "rmc"])
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument("--prefetch", default="nearest", choices=["none", "parent", "nearest"],
                   help="which already-computed drift the next round reuses")
    p.add_argument("--evaluate-leaves", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="evaluate the leaf level too (K^L extra rows per round); "
                        "only has an effect under --prefetch nearest")
    p.add_argument("--match", default="verification",
                   choices=["verification", "budget"],
                   help="what the RMC chain is matched on: the tree's target "
                        "batch (default) or its proposal budget B (the paper's)")
    p.add_argument("--n-workers", type=int, default=1,
                   help="parallel worker processes; 1 disables multiprocessing. "
                        "Results are identical at any worker count -- streams are "
                        "keyed by (K, L, replicate), not by evaluation order. "
                        "Under --no-lazy, memory scales with workers: each holds "
                        "its own copy of the current cell's tree and states.")
    p.add_argument("--lazy", action=argparse.BooleanOptionalAction, default=False,
                   help="use the cost simulator (experiments/lazy.py) instead of "
                        "the eager sampler: realises only the committed branch, "
                        "so the top corner of the grid becomes runnable")
    p.add_argument("--max-verification-budget", type=int, default=60000,
                   help="skip cells whose target batch |I| exceeds this many rows")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.csv"

    # Persist the configuration so resumed runs reject incompatible settings.
    config = {k: v for k, v in sorted(vars(args).items()) if k != "out"}
    config_path = out_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        differing = {k: (previous.get(k), v) for k, v in config.items()
                     if k not in ("K_values", "L_values", "rules") and previous.get(k) != v}
        if differing:
            raise SystemExit(
                f"{config_path} was written by a different configuration:\n"
                + "\n".join(f"  {k}: {was!r} -> {now!r}" for k, (was, now) in differing.items())
                + "\n\nAppending would mix protocols in one raw.csv. Use a new --out, "
                  "or delete the directory to start over."
            )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    done = set()
    if raw_path.exists():
        with raw_path.open() as fh:
            for row in csv.DictReader(fh):
                done.add((row["rule"], int(row["K"]), int(row["L"])))
        print(f"resuming: {len(done)} cells already in {raw_path}")

    setting = models.build(args.dimension, args.num_components, args.num_steps,
                           args.eps, mixture_seed=args.mixture_seed)
    print(f"d={args.dimension} components={args.num_components} T={args.num_steps} "
          f"eps={args.eps} prefetch={args.prefetch} "
          f"evaluate_leaves={args.evaluate_leaves} match={args.match}")
    print(f"speculative steps {setting.num_steps} (dropped deterministic "
          f"{setting.deterministic_steps}); baseline = {setting.total_steps} NFEs\n")

    workers = max(1, args.n_workers)
    pool = None
    if workers > 1:
        # Set before the pool starts so children inherit it: NumPy's BLAS would
        # otherwise spawn its own threads per worker and oversubscribe the cores,
        # which is slower than running serially.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                            initargs=(config,))
        print(f"{workers} workers\n")

    new = not raw_path.exists()
    try:
        with raw_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            for L in args.L_values:
                for K in args.K_values:
                    verif = lazy.verification_budget(K, L)
                    # The cap exists because the *eager* sampler materialises
                    # every node; the simulator never does, so it does not apply
                    # there.
                    if not args.lazy and verif > args.max_verification_budget:
                        print(f"  skip  K={K} L={L}  |I|={verif:,} > "
                              f"--max-verification-budget")
                        continue
                    for rule in args.rules:
                        if (rule, K, L) in done:
                            continue
                        t0 = time.time()
                        jobs = [(rule, K, L, r) for r in range(args.replicates)]
                        if pool is None:
                            sampler = tree = None
                            if not args.lazy:
                                sampler, tree, _ = build_sampler(
                                    setting, rule, K, L, args.prefetch,
                                    args.evaluate_leaves, args.match)
                            rows = [one_trajectory(setting, *j[:3], j[3], config,
                                                   sampler, tree) for j in jobs]
                        else:
                            # chunked so a worker reuses its cached sampler
                            rows = pool.map(_job, jobs,
                                            chunksize=max(1, len(jobs) // workers))
                        writer.writerows(rows)
                        fh.flush()
                        os.fsync(fh.fileno())
                        calls = float(np.mean([r["target_calls"] for r in rows]))
                        budget = sum(K**i for i in range(1, L + 1))
                        print(f"  {rule:>5} K={K} L={L} B={budget:<9,} |I|={verif:<7,}"
                              f" calls={calls:6.2f}  {setting.num_steps/calls:5.2f}x"
                              f"  [{time.time()-t0:5.1f}s]")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"\nwrote {raw_path}")


if __name__ == "__main__":
    main()
