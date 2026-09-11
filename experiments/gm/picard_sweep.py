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
refinement_summary.csv
    Online node summaries by trajectory, round, Picard sweep, and tree depth.
eps<eps>/K<K>_L<L>/<rule>/J<J>/samples.npz
    Initial, terminal, and complete committed trajectories, keyed by replicate.

Each ``(eps, K, L, rule, J)`` cell has its own directory and is written
atomically, so a later run can add a rule to an existing output directory.
Replicates are checkpointed individually while a cell is running, so interrupted
runs resume without retaining every replicate in memory.
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
from experiments.verifier_config import configured_verifier, parse_verifier_options

from specdiff import (  # noqa: E402
    DelayedDriftProposal,
    DraftTree,
    RefinementRequest,
    RefinementUpdate,
    SpeculativeSampler,
    Verifier,
    create_verifier,
    available_verifiers,
    picard_drift_update_fn,
    picard_update_fn,
)

SCHEMA_VERSION = 4
RULE_DIRECTORY_SCHEMA = 4
"""First schema with one directory per rule; older runs cannot be resumed."""
PICARD_UPDATES = {"drift": picard_drift_update_fn, "increment": picard_update_fn}
"""``--picard-update`` choices. Configs saved before the flag used ``increment``."""
IDENTITY_FIELDS = (
    "trajectory_id",
    "eps",
    "rule",
    "K",
    "L",
    "J",
    "replicate",
    "match",
    "evaluate_leaves",
)
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
    "allocated_verification_budget",
    "actual_proposal_budget",
    "verification_budget",
    "chain_depth",
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
    "end_step",
    "lookahead",
    "cumulative_committed",
    "cumulative_target_calls",
    "cumulative_target_states_evaluated",
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
REFINEMENT_METRICS = (
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
REFINEMENT_STATISTICS = (
    "mean", "std", "rms", "min", "max", "p50", "p90", "p99",
    "zero_fraction",
)
REFINEMENT_FIELDS = IDENTITY_FIELDS + (
    "round_index",
    "sweep_index",
    "refinement_iteration",
    "node_depth",
    "node_count",
) + tuple(
    f"{metric}_{statistic}"
    for metric in REFINEMENT_METRICS
    for statistic in REFINEMENT_STATISTICS
)

TABLES = {
    "trajectories.csv": TRAJECTORY_FIELDS,
    "rounds.csv": ROUND_FIELDS,
    "levels.csv": LEVEL_FIELDS,
    "refinement_summary.csv": REFINEMENT_FIELDS,
}


PROGRESS_BAR_WIDTH = 24


def _elapsed(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


class ConfigurationProgress:
    """Dependency-free progress for the complete configuration grid."""

    def __init__(self, total: int, mode="auto", stream=None) -> None:
        self.total = int(total)
        self.done = 0
        self.stream = sys.stderr if stream is None else stream
        self.mode = self._resolve_mode(mode)
        self.started = time.time()
        self.last_plain = 0.0
        self.drawn = False

    def _resolve_mode(self, mode: str) -> str:
        if mode != "auto":
            return mode
        try:
            return "bar" if self.stream.isatty() else "plain"
        except Exception:  # noqa: BLE001
            return "plain"

    def update(self, advance=0, *, label="", force=False) -> None:
        self.done = min(self.total, self.done + int(advance))
        if self.mode == "none":
            return
        now = time.time()
        if (
            self.mode == "plain"
            and not force
            and self.done < self.total
            and now - self.last_plain < 60.0
        ):
            return
        elapsed = now - self.started
        rate = self.done / elapsed if self.done and elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate else 0.0
        fraction = self.done / max(self.total, 1)
        filled = min(PROGRESS_BAR_WIDTH, int(PROGRESS_BAR_WIDTH * fraction))
        bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
        eta = _elapsed(remaining) if rate else "--:--"
        suffix = f"  {label}" if label else ""
        line = (
            f"[{bar}] {self.done}/{self.total} configurations "
            f"({100.0 * fraction:5.1f}%) elapsed {_elapsed(elapsed)} ETA {eta}{suffix}"
        )
        if self.mode == "bar":
            self.stream.write("\r" + line + "\033[K")
            self.drawn = True
        else:
            self.stream.write(line + "\n")
            self.last_plain = now
        self.stream.flush()

    def log(self, message: str) -> None:
        """Print a message without leaving it embedded in the live bar."""
        if self.mode == "bar" and self.drawn:
            self.stream.write("\r\033[K")
        self.stream.write(message + "\n")
        self.stream.flush()
        self.drawn = False

    def close(self) -> None:
        if self.mode == "bar" and self.drawn:
            self.stream.write("\n")
            self.stream.flush()
        self.drawn = False


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


def _row_l2(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = array.reshape(len(array), -1)
    return np.sqrt(np.einsum("ij,ij->i", flat, flat))


def _aggregate_values(metric: str, values) -> dict:
    if values is None:
        return {
            f"{metric}_{statistic}": ""
            for statistic in REFINEMENT_STATISTICS
        }
    array = np.asarray(values, dtype=float)
    quantiles = np.percentile(array, (50, 90, 99))
    return {
        f"{metric}_mean": float(np.mean(array)),
        f"{metric}_std": float(np.std(array)),
        f"{metric}_rms": float(np.sqrt(np.mean(array * array))),
        f"{metric}_min": float(np.min(array)),
        f"{metric}_max": float(np.max(array)),
        f"{metric}_p50": float(quantiles[0]),
        f"{metric}_p90": float(quantiles[1]),
        f"{metric}_p99": float(quantiles[2]),
        f"{metric}_zero_fraction": float(np.mean(array == 0.0)),
    }


class RecordingPicardUpdate:
    """Picard callback with online summaries by round/sweep/depth."""

    def __init__(self, tree: DraftTree, update_fn=picard_drift_update_fn) -> None:
        self.tree = tree
        self.update_fn = update_fn
        self.summaries: list[dict] = []
        self.sweep_index = 0
        self.round_index = -1
        self._previous_states = None

    def reset(self) -> None:
        self.summaries = []
        self.sweep_index = 0
        self.round_index = -1
        self._previous_states = None

    def __call__(self, request: RefinementRequest) -> RefinementUpdate:
        if request.iteration == 0:
            self.round_index += 1
            self._previous_states = None
        update = self.update_fn(request)
        states = np.asarray(request.parent_states)
        current = np.asarray(request.current_proposal_means)
        target = np.asarray(update.exact_target_means)
        # m(X) - X at the snapshot, whichever form the update carries.
        increment = target - states
        mismatch_l2 = _row_l2(target - current)
        state_size = int(np.prod(states.shape[1:]))
        change_l2 = (
            None
            if self._previous_states is None
            else _row_l2(states - self._previous_states)
        )
        metrics = {
            "state_l2": _row_l2(states),
            "current_proposal_mean_l2": _row_l2(current),
            "target_mean_l2": _row_l2(target),
            "target_drift_l2": _row_l2(increment),
            "picard_increment_l2": _row_l2(increment),
            "current_mean_mismatch_l2": mismatch_l2,
            "current_delta": mismatch_l2 / np.asarray(request.sigmas),
            "iterate_change_l2": change_l2,
            "iterate_change_rms": (
                None if change_l2 is None else change_l2 / math.sqrt(state_size)
            ),
        }
        depths = np.fromiter(
            (self.tree.depth_of(int(node)) for node in request.nodes),
            dtype=int,
            count=len(request.nodes),
        )
        for depth in np.unique(depths):
            selected = depths == depth
            summary = {
                "round_index": self.round_index,
                "sweep_index": self.sweep_index,
                "refinement_iteration": int(request.iteration) + 1,
                "node_depth": int(depth),
                "node_count": int(np.sum(selected)),
            }
            for metric, values in metrics.items():
                summary.update(
                    _aggregate_values(
                        metric, None if values is None else values[selected]
                    )
                )
            self.summaries.append(summary)
        self._previous_states = np.array(states, copy=True)
        self.sweep_index += 1
        return update


def trajectory_rngs(seed: int, replicate: int):
    """Common random numbers across every configuration."""

    init, run = np.random.SeedSequence(seed, spawn_key=(replicate,)).spawn(2)
    return np.random.default_rng(init), np.random.default_rng(run)


def matched_chain_depth(
    tree: DraftTree,
    num_steps: int,
    match: str,
    evaluate_leaves: bool,
) -> int:
    """Return the horizon-clamped RMC depth matched to ``tree``."""
    return create_verifier("rmc").matched_tree(
        tree, num_steps=num_steps, match=match, evaluate_leaves=evaluate_leaves
    ).depth


def make_tree(rule, K, L, num_steps, match, evaluate_leaves) -> DraftTree:
    uniform = DraftTree.uniform(K, L)
    return create_verifier(rule).matched_tree(
        uniform, num_steps=num_steps, match=match, evaluate_leaves=evaluate_leaves
    )


def epsilon_slug(eps: float) -> str:
    return f"eps{eps:g}"


def topology_slug(K: int, L: int) -> str:
    return f"K{K}_L{L}"


def iteration_slug(J: int) -> str:
    return f"J{J}"


def cell_slug(eps: float, rule: str, K: int, L: int, J: int) -> str:
    return f"{epsilon_slug(eps)}-{rule.replace('-', '_')}-{topology_slug(K, L)}-{iteration_slug(J)}"


def build_sampler(setting, rule, K, L, J, cfg):
    allocated_tree = DraftTree.uniform(K, L)
    tree = make_tree(
        rule, K, L, setting.num_steps, cfg["match"], cfg["evaluate_leaves"]
    )
    verifier = RecordingVerifier(configured_verifier(rule, cfg.get("verifier_options")))
    refiner = RecordingPicardUpdate(tree, PICARD_UPDATES[cfg.get("picard_update", "drift")])
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
    return sampler, verifier, refiner, tree, allocated_tree


def _identity(rule, K, L, J, replicate, cfg):
    return {
        "trajectory_id": f"{cell_slug(cfg['eps'], rule, K, L, J)}-r{replicate:06d}",
        "eps": cfg["eps"],
        "rule": rule,
        "K": K,
        "L": L,
        "J": J,
        "replicate": replicate,
        "match": cfg["match"],
        "evaluate_leaves": int(cfg["evaluate_leaves"]),
    }


def one_trajectory(setting, rule, K, L, J, replicate, cfg, built):
    sampler, verifier, refiner, tree, allocated_tree = built
    init_rng, run_rng = trajectory_rngs(cfg["seed"], replicate)
    initial = setting.initial_state(init_rng)
    refiner.reset()
    started = time.perf_counter()
    result = sampler.sample(initial, rng=run_rng)
    sampling_seconds = time.perf_counter() - started
    identity = _identity(rule, K, L, J, replicate, cfg)

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
        "B": allocated_tree.budget,
        "allocated_verification_budget": allocated_tree.verification_budget(
            evaluate_leaves=cfg["evaluate_leaves"]
        ),
        "actual_proposal_budget": tree.budget,
        "verification_budget": tree.verification_budget(
            evaluate_leaves=cfg["evaluate_leaves"]
        ),
        "chain_depth": tree.depth if create_verifier(rule).requires_chain else "",
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
    cumulative_target_calls = 0
    cumulative_target_states = 0
    for round_index, record in enumerate(result.rounds):
        cumulative_target_calls += record.target_calls
        cumulative_target_states += record.target_states_evaluated
        row = {
            **identity,
            "round_index": round_index,
            "start_step": record.start_step,
            "end_step": record.start_step + record.committed,
            "lookahead": record.lookahead,
            "cumulative_committed": record.start_step + record.committed,
            "cumulative_target_calls": cumulative_target_calls,
            "cumulative_target_states_evaluated": cumulative_target_states,
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

    refinement_rows = [
        {**identity, **summary}
        for summary in refiner.summaries
    ]

    return {
        "trajectory": trajectory_row,
        "rounds": round_rows,
        "levels": level_rows,
        "refinement_summary": refinement_rows,
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


def _job_trajectory(args):
    rule, K, L, J, replicate = args
    cell = (K, L, J)
    if _WORKER.get("cell") != cell:
        _WORKER["samplers"].clear()
        _WORKER["cell"] = cell
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


def _bundle_rows(filename, bundles):
    key = filename.removesuffix(".csv")
    if key == "trajectories":
        return [bundle["trajectory"] for bundle in bundles]
    return [row for bundle in bundles for row in bundle[key]]


def _sample_arrays(bundles):
    return {
        "replicate": np.array([b["trajectory"]["replicate"] for b in bundles]),
        "trajectory_id": np.array([
            b["trajectory"]["trajectory_id"] for b in bundles
        ]),
        "eps": np.array([b["trajectory"]["eps"] for b in bundles]),
        "rule": np.array([b["trajectory"]["rule"] for b in bundles]),
        "K": np.array([b["trajectory"]["K"] for b in bundles]),
        "L": np.array([b["trajectory"]["L"] for b in bundles]),
        "J": np.array([b["trajectory"]["J"] for b in bundles]),
        "match": np.array([b["trajectory"]["match"] for b in bundles]),
        "evaluate_leaves": np.array([
            b["trajectory"]["evaluate_leaves"] for b in bundles
        ]),
        "initial": np.stack([b["initial"] for b in bundles]),
        "sample": np.stack([b["sample"] for b in bundles]),
        "trajectory": np.stack([b["path"] for b in bundles]),
    }


def _write_samples(path: Path, arrays) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bundle_files(directory, bundles, save_samples) -> None:
    for filename, fields in TABLES.items():
        _write_csv(
            directory / filename, fields, _bundle_rows(filename, bundles)
        )
    if save_samples:
        _write_samples(directory / "samples.npz", _sample_arrays(bundles))


def write_cell(out: Path, rule, K, L, J, bundles, save_samples=True) -> Path:
    """Write a complete cell atomically; retained for small callers/tests."""
    parent = out / topology_slug(K, L) / rule
    parent.mkdir(parents=True, exist_ok=True)
    slug = iteration_slug(J)
    destination = parent / slug
    temporary = parent / f".{slug}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    _write_bundle_files(temporary, bundles, save_samples)
    (temporary / "COMPLETE").write_text("ok\n")
    if destination.exists():
        shutil.rmtree(temporary)
        return destination
    os.replace(temporary, destination)
    return destination


def streamed_cell_paths(out: Path, rule: str, K: int, L: int, J: int):
    parent = out / topology_slug(K, L) / rule
    parent.mkdir(parents=True, exist_ok=True)
    slug = iteration_slug(J)
    return parent / slug, parent / f".{slug}.work"


def completed_replicates(work: Path) -> set[int]:
    replicates = work / "replicates"
    if not replicates.exists():
        return set()
    return {
        int(marker.parent.name[1:])
        for marker in replicates.glob("r*/COMPLETE")
    }


def write_replicate(work: Path, replicate: int, bundles, save_samples=True) -> Path:
    replicates = work / "replicates"
    replicates.mkdir(parents=True, exist_ok=True)
    slug = f"r{replicate:06d}"
    destination = replicates / slug
    if (destination / "COMPLETE").exists():
        return destination
    temporary = replicates / f".{slug}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    _write_bundle_files(temporary, bundles, save_samples)
    (temporary / "COMPLETE").write_text("ok\n")
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    return destination


def finalize_streamed_cell(
    work: Path,
    destination: Path,
    expected_replicates: int,
    save_samples=True,
) -> Path:
    shards = sorted((work / "replicates").glob("r*"))
    complete = [shard for shard in shards if (shard / "COMPLETE").exists()]
    if len(complete) != expected_replicates:
        raise RuntimeError(
            f"cannot finalize {destination}: {len(complete)}/"
            f"{expected_replicates} replicates are complete"
        )
    for filename, fields in TABLES.items():
        temporary = work / f".{filename}.tmp"
        with temporary.open("w", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            for shard in complete:
                with (shard / filename).open(newline="") as src:
                    writer.writerows(csv.DictReader(src))
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, work / filename)
    if save_samples:
        combined = {}
        for shard in complete:
            with np.load(shard / "samples.npz") as archive:
                for field in archive.files:
                    combined.setdefault(field, []).append(np.array(archive[field]))
        _write_samples(
            work / "samples.npz",
            {field: np.concatenate(parts) for field, parts in combined.items()},
        )
    (work / "COMPLETE").write_text("ok\n")
    if destination.exists():
        raise RuntimeError(f"incomplete destination already exists: {destination}")
    os.replace(work, destination)
    shards = destination / "replicates"
    if shards.exists():
        shutil.rmtree(shards)
    return destination


def consolidate(out: Path) -> None:
    """Rebuild normalized epsilon-level tables from every completed cell.

    The glob covers rules added by earlier invocations as well as this one.
    """
    complete = sorted(marker.parent for marker in out.glob("K*_L*/*/J*/COMPLETE"))
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
    p.add_argument(
        "--eps-values", "--eps", dest="eps_values", type=float, nargs="+",
        default=[0.1, 0.3, 0.6], help="churn values; each gets an eps<value> directory",
    )
    p.add_argument("--mixture-seed", type=int, default=20260714)
    p.add_argument("--seed", type=int, default=20260714)
    p.add_argument("--K-values", type=int, nargs="+", default=list(range(1, 8)))
    p.add_argument("--L-values", type=int, nargs="+", default=list(range(1, 8)))
    refinement_grid = p.add_mutually_exclusive_group()
    refinement_grid.add_argument(
        "--J-values", type=int, nargs="+", default=argparse.SUPPRESS
    )
    refinement_grid.add_argument(
        "--J-up-to-L",
        action="store_true",
        default=argparse.SUPPRESS,
        help="for each depth L, sweep Picard iterations J from 0 through L",
    )
    p.add_argument(
        "--picard-update", default="drift", choices=sorted(PICARD_UPDATES),
        help="freeze the target's drift (default) or the whole increment m - x; "
        "runs saved before this flag used increment",
    )
    p.add_argument("--rules", nargs="+", default=["rmc", "d-grs", "paws"])
    p.add_argument("--verifier-options", type=parse_verifier_options, default={})
    p.add_argument(
        "--match", default="verification", choices=["verification", "budget"],
        help="match each RMC chain to the (K,L) tree's target batch or proposal budget",
    )
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
    p.add_argument(
        "--progress", default="auto", choices=["auto", "bar", "plain", "none"],
        help="overall configuration progress display",
    )
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument(
        "--max-verification-budget", type=int, default=0,
        help="skip larger (K,L) trees; 0 keeps the complete canonical grid",
    )
    return p


def picard_iterations(args, L: int):
    """Return the Picard-iteration grid for a tree of depth ``L``."""
    if hasattr(args, "J_values"):
        return args.J_values
    return range(L + 1)


def configuration_count(args) -> int:
    """Count ``(eps, rule, K, L, J)`` configurations in this run."""
    per_epsilon = len(args.rules) * len(args.K_values) * sum(
        len(tuple(picard_iterations(args, L))) for L in args.L_values
    )
    return len(args.eps_values) * per_epsilon


def _run_epsilon(
    args, out: Path, cfg: dict, eps: float, progress: ConfigurationProgress
) -> None:
    eps_out = out / epsilon_slug(eps)
    eps_out.mkdir(parents=True, exist_ok=True)
    eps_cfg = {**cfg, "eps": eps}
    (eps_out / "config.json").write_text(
        json.dumps(eps_cfg, indent=2, sort_keys=True) + "\n"
    )
    setting = models.build(
        args.dimension,
        args.num_components,
        args.num_steps,
        eps,
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
            workers, initializer=_init_worker, initargs=(eps_cfg,)
        )

    progress.log(
        f"eps={eps:g} d={args.dimension} components={args.num_components} "
        f"T={args.num_steps}; speculative steps={setting.num_steps} "
        f"match={args.match} evaluate_leaves={args.evaluate_leaves}"
    )
    try:
        for K in args.K_values:
            for L in args.L_values:
                iterations = tuple(picard_iterations(args, L))
                allocated_tree = DraftTree.uniform(K, L)
                budget = allocated_tree.verification_budget(
                    evaluate_leaves=args.evaluate_leaves
                )
                if args.max_verification_budget and budget > args.max_verification_budget:
                    label = f"{epsilon_slug(eps)}/{topology_slug(K, L)} skipped"
                    progress.log(f"skip {label}: |I|={budget:,} exceeds cap")
                    progress.update(len(iterations) * len(args.rules), label=label)
                    continue
                for J in iterations:
                    pending = {}
                    for rule in args.rules:
                        destination, work = streamed_cell_paths(
                            eps_out, rule, K, L, J
                        )
                        if (destination / "COMPLETE").exists():
                            label = str(destination.relative_to(out))
                            progress.log(f"resume: {label} already complete")
                            progress.update(1, label=f"{label} resumed")
                            continue
                        work.mkdir(exist_ok=True)
                        completed = completed_replicates(work)
                        pending[rule] = (destination, work, [
                            replicate for replicate in range(args.replicates)
                            if replicate not in completed
                        ])
                    if not pending:
                        continue
                    # One ordered stream for every pending rule keeps the pool
                    # busy across rule boundaries; results arrive rule by rule.
                    jobs = [
                        (rule, replicate)
                        for rule, (_, _, missing) in pending.items()
                        for replicate in missing
                    ]
                    if pool is not None:
                        results = pool.imap(
                            _job_trajectory,
                            [(rule, K, L, J, replicate) for rule, replicate in jobs],
                            chunksize=1,
                        )
                    else:
                        built = {
                            rule: build_sampler(setting, rule, K, L, J, eps_cfg)
                            for rule in pending
                        }
                        results = (
                            one_trajectory(
                                setting, rule, K, L, J, replicate,
                                eps_cfg, built[rule],
                            )
                            for rule, replicate in jobs
                        )
                    for rule, (destination, work, missing) in pending.items():
                        label = str(destination.relative_to(out))
                        done = args.replicates - len(missing)
                        progress.update(
                            label=f"{label} replicate {done}/{args.replicates}"
                        )
                        started = time.time()
                        for finished, replicate in enumerate(missing, start=done + 1):
                            write_replicate(
                                work, replicate, [next(results)],
                                save_samples=args.save_samples,
                            )
                            progress.update(
                                label=f"{label} replicate {finished}/{args.replicates}"
                            )
                        finalize_streamed_cell(
                            work, destination, args.replicates,
                            save_samples=args.save_samples,
                        )
                        with (destination / "trajectories.csv").open(newline="") as src:
                            trajectory_rows = list(csv.DictReader(src))
                        mean_calls = float(np.mean([
                            float(row["target_calls"]) for row in trajectory_rows
                        ]))
                        mean_accept = float(np.mean([
                            float(row["acceptance_rate"]) for row in trajectory_rows
                        ]))
                        progress.log(
                            f"  {rule:>5} {topology_slug(K, L)}/{iteration_slug(J)} "
                            f"calls={mean_calls:6.2f} "
                            f"speed={setting.num_steps / mean_calls:5.2f}x "
                            f"accept={mean_accept:6.3f} "
                            f"[{time.time() - started:5.1f}s]"
                        )
                        progress.update(1, label=label)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    consolidate(eps_out)


def resumed_config(path: Path, cfg: dict) -> dict:
    """Merge this invocation's ``cfg`` into the run saved at ``path``.

    Each cell is its own ``(eps, K, L, rule, J)`` directory, so a new rule can
    join an existing run: the saved rule list grows, and a rule's verifier
    options are fixed by its first run. K and L may be subsets of the saved
    grid; every other setting must match.
    """
    previous = json.loads(path.read_text())
    if previous.get("schema_version", 0) < RULE_DIRECTORY_SCHEMA:
        raise SystemExit(
            f"{path} predates per-rule cell directories and cannot be resumed; "
            "use a new --out directory"
        )
    # These keys are reconciled below; every other key is the protocol.
    merged = {"schema_version", "K_values", "L_values", "rules", "verifier_options"}
    previous_protocol = {
        key: value for key, value in previous.items() if key not in merged
    }
    current_protocol = {
        key: value for key, value in cfg.items() if key not in merged
    }
    if previous_protocol != current_protocol:
        raise SystemExit(f"{path} has a different protocol; use a new --out directory")
    for axis in ("K_values", "L_values"):
        if not set(cfg[axis]).issubset(previous[axis]):
            raise SystemExit(
                f"{path}: {axis} must be a subset of the saved grid "
                f"{previous[axis]}; use a new --out directory to expand it"
            )
        # Keep the original grid in metadata: older cells remain valid,
        # and a later invocation may resume any part of that grid.
        # The run loops and progress count use args, the selected subset.
        cfg[axis] = previous[axis]
    saved_options = previous["verifier_options"]
    for rule in cfg["rules"]:
        options = cfg["verifier_options"].get(rule, {})
        if rule in previous["rules"] and options != saved_options.get(rule, {}):
            raise SystemExit(
                f"{path}: {rule} ran with verifier options "
                f"{saved_options.get(rule, {})}, not {options}; "
                "use a new --out directory for a variant"
            )
    cfg["verifier_options"] = {**saved_options, **cfg["verifier_options"]}
    # Saved rules stay listed even when this invocation runs only new ones;
    # their cells remain valid and the consolidated tables include them.
    cfg["rules"] = previous["rules"] + [
        rule for rule in cfg["rules"] if rule not in previous["rules"]
    ]
    return cfg


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if any(x < 1 for x in args.K_values + args.L_values):
        raise SystemExit("K and L values must be positive")
    if hasattr(args, "J_values") and any(x < 0 for x in args.J_values):
        raise SystemExit("J values must be non-negative")
    if any(eps < 0 for eps in args.eps_values):
        raise SystemExit("eps values must be non-negative")
    eps_slugs = [epsilon_slug(eps) for eps in args.eps_values]
    if len(set(eps_slugs)) != len(eps_slugs):
        raise SystemExit("eps values must map to distinct output-directory names")
    if len(set(args.rules)) != len(args.rules):
        raise SystemExit("rules must be distinct")
    unknown = set(args.rules) - set(available_verifiers())
    if unknown:
        raise SystemExit(f"unsupported rules: {sorted(unknown)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in {"out", "n_workers", "progress"}
    }
    cfg["schema_version"] = SCHEMA_VERSION
    # Options belong to the rule they configure: record only the rules that run.
    cfg["verifier_options"] = {
        rule: options for rule, options in args.verifier_options.items()
        if rule in args.rules
    }
    config_path = out / "config.json"
    if config_path.exists():
        cfg = resumed_config(config_path, cfg)
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    (out / "schema.json").write_text(json.dumps({
        "version": SCHEMA_VERSION,
        "tables": {name: list(fields) for name, fields in TABLES.items()},
        "samples": (
            "eps<eps>/K<K>_L<L>/<rule>/J<J>/samples.npz: replicate, trajectory_id, "
            "eps, rule, K, L, J, match, evaluate_leaves, initial, sample, trajectory"
        ),
    }, indent=2, sort_keys=True) + "\n")

    progress = ConfigurationProgress(
        configuration_count(args), mode=args.progress
    )
    progress.update(force=True, label="starting")
    try:
        for eps in args.eps_values:
            _run_epsilon(args, out, cfg, eps, progress)
        progress.update(force=True, label="complete")
    finally:
        progress.close()
    print(f"wrote normalized tables under each epsilon directory in {out}")

if __name__ == "__main__":
    main()
