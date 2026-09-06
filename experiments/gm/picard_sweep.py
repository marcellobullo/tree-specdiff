"""Eager Gaussian-mixture experiments for synchronous tree Picard refinement.

Unlike gm_sweep, this runner makes refinement count J a first-class axis and
records enough normalized data to support new plots without rerunning the
sampler. Picard refinement needs every internal tree node, so the lazy branch
simulator is intentionally not supported.

Outputs
-------
trajectories.csv
    One aggregate row per sampled trajectory.
rounds.csv
    One row per speculative round, including the full target-cost decomposition.
levels.csv
    One row per verification decision, including normalized mean mismatch.
refinements.csv
    One row per internal node and Picard sweep, including iterate change.
cells/<cell>/samples.npz
    Initial, terminal, and complete committed trajectories, keyed by replicate.

Each cell is written atomically before the four top-level CSVs are rebuilt.
Interrupted runs therefore resume at cell granularity without mixing partial
tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.gm.models as models  # noqa: E402

from specdiff import (  # noqa: E402
    DelayedDriftProposal,
    DraftTree,
    RefinementRequest,
    RefinementUpdate,
    SpeculativeSampler,
    Verifier,
    create_verifier,
    picard_update_fn,
)

SCHEMA_VERSION = 1
IDENTITY_FIELDS = ("trajectory_id", "rule", "K", "L", "J", "replicate")
COST_FIELDS = (
    "proposal_target_calls",
    "proposal_target_states_evaluated",
    "refinement_target_calls",
    "refinement_target_states_evaluated",
    "verification_target_calls",
    "verification_target_states_evaluated",
    "verification_target_means_reused",
    "target_calls",
    "target_states_evaluated",
)
TRAJECTORY_FIELDS = IDENTITY_FIELDS + (
    "B",
    "verification_budget",
    "dimension",
    "num_steps",
    "total_steps",
    "deterministic_steps",
    "rounds",
    "verified_levels",
    "accepted_depth",
    "rejected_rounds",
    "acceptance_rate",
    "mean_committed",
    "max_committed",
    "drafted_states",
    "proposals_examined",
    "mean_proposals_examined",
) + COST_FIELDS + (
    "speculative_speedup",
    "effective_total_target_calls",
    "end_to_end_speedup",
    "initial_l2",
    "initial_mean",
    "initial_std",
    "sample_l2",
    "sample_mean",
    "sample_std",
    "sample_min",
    "sample_max",
    "sampling_seconds",
)
ROUND_FIELDS = IDENTITY_FIELDS + (
    "round_index",
    "start_step",
    "lookahead",
    "committed",
    "accepted_depth",
    "rejected",
    "drafted",
    "verified",
    "proposals_examined",
) + COST_FIELDS
LEVEL_FIELDS = IDENTITY_FIELDS + (
    "round_index",
    "start_step",
    "round_lookahead",
    "level",
    "node",
    "step",
    "sigma",
    "num_children",
    "accepted",
    "rejected",
    "child_index",
    "proposals_examined",
    "guaranteed_picard_prefix",
    "mean_mismatch_l2",
    "mean_mismatch_rms",
    "delta",
    "proposal_drift_l2",
    "target_drift_l2",
    "drift_cosine",
    "parent_l2",
    "proposal_mean_l2",
    "target_mean_l2",
    "returned_state_l2",
    "returned_target_residual",
    "candidate_target_distance_min",
    "candidate_target_distance_mean",
    "candidate_target_distance_max",
)
REFINEMENT_FIELDS = IDENTITY_FIELDS + (
    "round_index",
    "sweep_index",
    "refinement_iteration",
    "node",
    "node_depth",
    "step",
    "sigma",
    "state_l2",
    "current_proposal_mean_l2",
    "target_mean_l2",
    "target_drift_l2",
    "picard_increment_l2",
    "current_mean_mismatch_l2",
    "current_delta",
    "iterate_change_l2",
    "iterate_change_rms",
)

TABLES = {
    "trajectories.csv": TRAJECTORY_FIELDS,
    "rounds.csv": ROUND_FIELDS,
    "levels.csv": LEVEL_FIELDS,
    "refinements.csv": REFINEMENT_FIELDS,
}


def _l2(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float).ravel()))


def _rms(x) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def _cosine(x, y) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else math.nan


class RecordingVerifier(Verifier):
    """Transparent verifier wrapper that records final proposal diagnostics."""

    def __init__(self, inner: Verifier) -> None:
        self.inner = inner
        self.name = f"recording({inner.name})"
        self.max_children = inner.max_children
        self.events: list[dict] = []

    def reset(self) -> None:
        self.events = []
        self.inner.reset()

    def check_topology(self, tree) -> None:
        self.inner.check_topology(tree)

    def verify(self, request):
        result = self.inner.verify(request)
        proposal = np.asarray(request.proposal_mean)
        target = np.asarray(request.target_mean)
        parent = np.asarray(request.parent_state)
        mismatch = target - proposal
        proposal_drift = proposal - parent
        target_drift = target - parent
        candidates = np.asarray(request.children)
        distances = np.linalg.norm(
            (candidates - target).reshape(len(candidates), -1), axis=1
        ) / request.sigma
        self.events.append({
            "level": int(request.info.get("level", 0)),
            "node": int(request.info.get("node", 0)),
            "step": int(request.step),
            "sigma": float(request.sigma),
            "num_children": int(request.num_children),
            "accepted": int(result.accepted),
            "rejected": int(not result.accepted),
            "child_index": "" if result.child_index is None else int(result.child_index),
            "proposals_examined": (
                "" if result.proposals_examined is None else int(result.proposals_examined)
            ),
            "mean_mismatch_l2": _l2(mismatch),
            "mean_mismatch_rms": _rms(mismatch),
            "delta": _l2(mismatch) / request.sigma,
            "proposal_drift_l2": _l2(proposal_drift),
            "target_drift_l2": _l2(target_drift),
            "drift_cosine": _cosine(proposal_drift, target_drift),
            "parent_l2": _l2(parent),
            "proposal_mean_l2": _l2(proposal),
            "target_mean_l2": _l2(target),
            "returned_state_l2": _l2(result.state),
            "returned_target_residual": _l2(np.asarray(result.state) - target)
            / request.sigma,
            "candidate_target_distance_min": float(distances.min()),
            "candidate_target_distance_mean": float(distances.mean()),
            "candidate_target_distance_max": float(distances.max()),
        })
        return result


class RecordingPicardUpdate:
    """Canonical Picard callback with one diagnostic row per internal node."""

    def __init__(self, tree: DraftTree) -> None:
        self.tree = tree
        self.events: list[dict] = []
        self.sweep_index = 0
        self._previous: dict[tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        self.events = []
        self.sweep_index = 0
        self._previous = {}

    def __call__(self, request: RefinementRequest) -> RefinementUpdate:
        if request.iteration == 0:
            self._previous = {}
        update = picard_update_fn(request)
        states = np.asarray(request.parent_states)
        current = np.asarray(request.current_proposal_means)
        target = np.asarray(update.exact_target_means)
        for row, (image, node, step, sigma) in enumerate(zip(
            request.indices_in_batch,
            request.nodes,
            request.steps,
            request.sigmas,
        )):
            state = states[row]
            key = (int(image), int(node))
            previous = self._previous.get(key)
            change = None if previous is None else state - previous
            mismatch = target[row] - current[row]
            increment = np.asarray(update.increments[row])
            self.events.append({
                "sweep_index": self.sweep_index,
                "refinement_iteration": int(request.iteration) + 1,
                "node": int(node),
                "node_depth": self.tree.depth_of(int(node)),
                "step": int(step),
                "sigma": float(sigma),
                "state_l2": _l2(state),
                "current_proposal_mean_l2": _l2(current[row]),
                "target_mean_l2": _l2(target[row]),
                "target_drift_l2": _l2(target[row] - state),
                "picard_increment_l2": _l2(increment),
                "current_mean_mismatch_l2": _l2(mismatch),
                "current_delta": _l2(mismatch) / sigma,
                "iterate_change_l2": "" if change is None else _l2(change),
                "iterate_change_rms": "" if change is None else _rms(change),
            })
            self._previous[key] = np.array(state, copy=True)
        self.sweep_index += 1
        return update


def trajectory_rngs(seed: int, replicate: int):
    """Common random numbers across every configuration."""

    init, run = np.random.SeedSequence(seed, spawn_key=(replicate,)).spawn(2)
    return np.random.default_rng(init), np.random.default_rng(run)


def make_tree(rule: str, K: int, L: int) -> DraftTree:
    if rule == "rmc":
        if K != 1:
            raise ValueError("RMC configurations require K=1")
        return DraftTree.chain(L)
    return DraftTree.uniform(K, L)


def cell_slug(rule: str, K: int, L: int, J: int) -> str:
    return f"{rule.replace('-', '_')}-K{K}-L{L}-J{J}"


def build_sampler(setting, rule, K, L, J, cfg):
    tree = make_tree(rule, K, L)
    verifier = RecordingVerifier(create_verifier(rule))
    refiner = RecordingPicardUpdate(tree)
    sampler = SpeculativeSampler(
        target=setting.target,
        proposal=DelayedDriftProposal(setting.target),
        schedule=setting.schedule,
        tree=tree,
        verifier=verifier,
        num_steps=setting.num_steps,
        prefetch=cfg["prefetch"],
        evaluate_leaves=cfg["evaluate_leaves"],
        check_contract=cfg["check_contract"],
        proposal_refinement_iters=J,
        refinement_update_fn=refiner,
    )
    return sampler, verifier, refiner, tree


def _identity(rule, K, L, J, replicate):
    return {
        "trajectory_id": f"{cell_slug(rule, K, L, J)}-r{replicate:06d}",
        "rule": rule,
        "K": K,
        "L": L,
        "J": J,
        "replicate": replicate,
    }


def one_trajectory(setting, rule, K, L, J, replicate, cfg, built):
    sampler, verifier, refiner, tree = built
    init_rng, run_rng = trajectory_rngs(cfg["seed"], replicate)
    initial = setting.initial_state(init_rng)
    refiner.reset()
    started = time.perf_counter()
    result = sampler.sample(initial, rng=run_rng)
    sampling_seconds = time.perf_counter() - started
    identity = _identity(rule, K, L, J, replicate)

    cost = {
        field: sum(getattr(record, field) for record in result.rounds)
        for field in COST_FIELDS
    }
    if cost["target_calls"] != result.target_calls:
        raise RuntimeError("round target-call accounting does not match result total")
    if cost["target_states_evaluated"] != result.target_states_evaluated:
        raise RuntimeError("round target-state accounting does not match result total")
    if cost["target_calls"] != (
        cost["proposal_target_calls"]
        + cost["refinement_target_calls"]
        + cost["verification_target_calls"]
    ):
        raise RuntimeError("target-call decomposition is inconsistent")

    committed = sum(r.committed for r in result.rounds)
    accepted = sum(r.accepted_depth for r in result.rounds)
    examined = [
        value
        for record in result.rounds
        for value in record.proposals_examined
        if value is not None
    ]
    deterministic = len(setting.deterministic_steps)
    effective_calls = result.target_calls + deterministic
    terminal = np.asarray(result.sample)
    trajectory_row = {
        **identity,
        "B": tree.budget,
        "verification_budget": tree.verification_budget(
            evaluate_leaves=cfg["evaluate_leaves"]
        ),
        "dimension": setting.dimension,
        "num_steps": setting.num_steps,
        "total_steps": setting.total_steps,
        "deterministic_steps": deterministic,
        "rounds": len(result.rounds),
        "verified_levels": committed,
        "accepted_depth": accepted,
        "rejected_rounds": sum(r.rejected for r in result.rounds),
        "acceptance_rate": accepted / committed,
        "mean_committed": committed / len(result.rounds),
        "max_committed": max(r.committed for r in result.rounds),
        "drafted_states": result.drafted_states,
        "proposals_examined": sum(examined),
        "mean_proposals_examined": float(np.mean(examined)) if examined else math.nan,
        **cost,
        "speculative_speedup": setting.num_steps / result.target_calls,
        "effective_total_target_calls": effective_calls,
        "end_to_end_speedup": setting.total_steps / effective_calls,
        "initial_l2": _l2(initial),
        "initial_mean": float(np.mean(initial)),
        "initial_std": float(np.std(initial)),
        "sample_l2": _l2(terminal),
        "sample_mean": float(np.mean(terminal)),
        "sample_std": float(np.std(terminal)),
        "sample_min": float(np.min(terminal)),
        "sample_max": float(np.max(terminal)),
        "sampling_seconds": sampling_seconds,
    }

    round_rows = []
    for round_index, record in enumerate(result.rounds):
        row = {
            **identity,
            "round_index": round_index,
            "start_step": record.start_step,
            "lookahead": record.lookahead,
            "committed": record.committed,
            "accepted_depth": record.accepted_depth,
            "rejected": int(record.rejected),
            "drafted": record.drafted,
            "verified": record.verified,
            "proposals_examined": json.dumps(record.proposals_examined),
        }
        row.update({field: getattr(record, field) for field in COST_FIELDS})
        round_rows.append(row)

    level_rows, cursor = [], 0
    for round_index, record in enumerate(result.rounds):
        events = verifier.events[cursor:cursor + record.committed]
        if len(events) != record.committed:
            raise RuntimeError("verification event count does not match round record")
        for event in events:
            level_rows.append({
                **identity,
                "round_index": round_index,
                "start_step": record.start_step,
                "round_lookahead": record.lookahead,
                **event,
                "guaranteed_picard_prefix": int(event["level"] <= J),
            })
        cursor += record.committed
    if cursor != len(verifier.events):
        raise RuntimeError("unassigned verification events remain")

    refinement_rows = []
    for event in refiner.events:
        round_index = event["sweep_index"] // J if J else 0
        refinement_rows.append({**identity, "round_index": round_index, **event})

    return {
        "trajectory": trajectory_row,
        "rounds": round_rows,
        "levels": level_rows,
        "refinements": refinement_rows,
        "initial": np.array(initial, copy=True),
        "sample": np.array(terminal, copy=True),
        "path": np.array(result.trajectory, copy=True),
    }


_WORKER: dict = {}


def _init_worker(cfg):
    _WORKER["cfg"] = cfg
    _WORKER["setting"] = models.build(
        cfg["dimension"],
        cfg["num_components"],
        cfg["num_steps"],
        cfg["eps"],
        mixture_seed=cfg["mixture_seed"],
    )
    _WORKER["samplers"] = {}


def _job(args):
    rule, K, L, J, replicate = args
    key = (rule, K, L, J)
    if key not in _WORKER["samplers"]:
        _WORKER["samplers"][key] = build_sampler(
            _WORKER["setting"], rule, K, L, J, _WORKER["cfg"]
        )
    return one_trajectory(
        _WORKER["setting"], rule, K, L, J, replicate,
        _WORKER["cfg"], _WORKER["samplers"][key],
    )


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def write_cell(out: Path, rule, K, L, J, bundles, save_samples=True) -> Path:
    cells = out / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    slug = cell_slug(rule, K, L, J)
    destination = cells / slug
    temporary = cells / f".{slug}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    for filename, fields in TABLES.items():
        key = filename.removesuffix(".csv")
        if key == "trajectories":
            rows = [bundle["trajectory"] for bundle in bundles]
        else:
            rows = [row for bundle in bundles for row in bundle[key]]
        _write_csv(temporary / filename, fields, rows)
    if save_samples:
        np.savez_compressed(
            temporary / "samples.npz",
            replicate=np.array([b["trajectory"]["replicate"] for b in bundles]),
            trajectory_id=np.array([b["trajectory"]["trajectory_id"] for b in bundles]),
            initial=np.stack([b["initial"] for b in bundles]),
            sample=np.stack([b["sample"] for b in bundles]),
            trajectory=np.stack([b["path"] for b in bundles]),
        )
    (temporary / "COMPLETE").write_text("ok\n")
    if destination.exists():
        shutil.rmtree(temporary)
        return destination
    os.replace(temporary, destination)
    return destination


def consolidate(out: Path) -> None:
    """Rebuild normalized top-level tables from atomically completed cells."""

    complete = sorted(
        path for path in (out / "cells").iterdir()
        if path.is_dir() and (path / "COMPLETE").exists()
    )
    for filename, fields in TABLES.items():
        temporary = out / f".{filename}.tmp"
        with temporary.open("w", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            for cell in complete:
                with (cell / filename).open(newline="") as src:
                    writer.writerows(csv.DictReader(src))
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, out / filename)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/gm-picard")
    p.add_argument("--dimension", type=int, default=512)
    p.add_argument("--num-components", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--eps", type=float, default=0.06)
    p.add_argument("--mixture-seed", type=int, default=20260714)
    p.add_argument("--seed", type=int, default=20260714)
    p.add_argument("--K-values", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--L-values", type=int, nargs="+", default=[4])
    p.add_argument("--J-values", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--rules", nargs="+", default=["rmc", "d-grs"])
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument(
        "--prefetch", default="nearest", choices=["none", "parent", "nearest"]
    )
    p.add_argument(
        "--evaluate-leaves", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--check-contract", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--save-samples", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--max-verification-budget", type=int, default=60000)
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if any(x < 1 for x in args.K_values + args.L_values):
        raise SystemExit("K and L values must be positive")
    if any(x < 0 for x in args.J_values):
        raise SystemExit("J values must be non-negative")
    unknown = set(args.rules) - {"rmc", "d-grs"}
    if unknown:
        raise SystemExit(f"unsupported rules: {sorted(unknown)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in sorted(vars(args).items()) if k not in {"out", "n_workers"}}
    cfg["schema_version"] = SCHEMA_VERSION
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise SystemExit(
            f"{config_path} has a different protocol; use a new --out directory"
        )
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    (out / "schema.json").write_text(json.dumps({
        "version": SCHEMA_VERSION,
        "tables": {name: list(fields) for name, fields in TABLES.items()},
        "samples": (
            "cells/<cell>/samples.npz: replicate, trajectory_id, initial, "
            "sample, trajectory"
        ),
    }, indent=2, sort_keys=True) + "\n")

    setting = models.build(
        args.dimension, args.num_components, args.num_steps, args.eps,
        mixture_seed=args.mixture_seed,
    )
    workers = max(1, args.n_workers)
    pool = None
    if workers > 1:
        for variable in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(variable, "1")
        pool = mp.get_context("spawn").Pool(
            workers, initializer=_init_worker, initargs=(cfg,)
        )

    print(
        f"d={args.dimension} components={args.num_components} T={args.num_steps} "
        f"eps={args.eps}; speculative steps={setting.num_steps}"
    )
    try:
        for rule in args.rules:
            for K in args.K_values:
                if rule == "rmc" and K != 1:
                    continue
                for L in args.L_values:
                    tree = make_tree(rule, K, L)
                    budget = tree.verification_budget(
                        evaluate_leaves=args.evaluate_leaves
                    )
                    if budget > args.max_verification_budget:
                        print(
                            f"skip {rule} K={K} L={L}: |I|={budget:,} exceeds cap"
                        )
                        continue
                    for J in args.J_values:
                        slug = cell_slug(rule, K, L, J)
                        destination = out / "cells" / slug
                        if (destination / "COMPLETE").exists():
                            print(f"resume: {slug} already complete")
                            continue
                        started = time.time()
                        jobs = [
                            (rule, K, L, J, replicate)
                            for replicate in range(args.replicates)
                        ]
                        if pool is not None:
                            bundles = pool.map(
                                _job, jobs, chunksize=max(1, len(jobs) // workers)
                            )
                        else:
                            built = build_sampler(setting, rule, K, L, J, cfg)
                            bundles = [
                                one_trajectory(
                                    setting, rule, K, L, J, replicate, cfg, built
                                )
                                for replicate in range(args.replicates)
                            ]
                        write_cell(
                            out, rule, K, L, J, bundles,
                            save_samples=args.save_samples,
                        )
                        consolidate(out)
                        mean_calls = float(np.mean([
                            b["trajectory"]["target_calls"] for b in bundles
                        ]))
                        mean_accept = float(np.mean([
                            b["trajectory"]["acceptance_rate"] for b in bundles
                        ]))
                        print(
                            f"{slug:>20} calls={mean_calls:6.2f} "
                            f"speed={setting.num_steps / mean_calls:5.2f}x "
                            f"accept={mean_accept:6.3f} "
                            f"[{time.time() - started:5.1f}s]"
                        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    consolidate(out)
    print(f"wrote normalized tables under {out}")


if __name__ == "__main__":
    main()
