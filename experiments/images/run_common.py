"""Shared, dependency-light bookkeeping for the EDM and SD3 experiment drivers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

SIGNATURE_VERSION = 1
_METRIC_KEYS = (
    "baseline_calls",
    "target_calls",
    "end_to_end_baseline_calls",
    "end_to_end_target_calls",
    "isolated_speedup_sum",
    "sample_count",
    "occupancy_active",
    "occupancy_slots",
    "accepted_levels",
    "verified_levels",
    "target_states_evaluated",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def file_identity(value: Optional[str]) -> Optional[dict]:
    """Cheap local-path identity suitable for detecting accidental reuse."""
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        return {"value": value}
    if path.is_file():
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return {"path": str(path.resolve()), "files": len(files),
            "manifest_sha256": digest.hexdigest()}


def run_signature(driver: str, args, setting, tree, *, extra: Mapping[str, Any]) -> dict:
    """Versioned signature for deciding whether a shard is safe to resume."""
    ignored = {"out", "device", "cpu", "no_accelerate", "overwrite", "check_contract"}
    config = {k: _plain(v) for k, v in vars(args).items() if k not in ignored}
    return {
        "version": SIGNATURE_VERSION,
        "driver": driver,
        "config": config,
        "total_steps": int(setting.total_steps),
        "speculative_steps": int(setting.num_steps),
        "state_shape": list(setting.state_shape),
        "tree_parents": [tree.parent(i) for i in range(tree.size)],
        "extra": _plain(dict(extra)),
    }


def metric_totals(result, *, num_steps: int, deterministic_steps: int) -> dict:
    """Raw additive counters from one batched sampling call."""
    accepted = sum(sum(r.accepted_depth) for r in result.rounds)
    verified = sum(sum(r.committed) for r in result.rounds)
    active = sum(len(r.active) for r in result.rounds)
    slots = len(result.rounds) * result.batch_size
    isolated = sum(num_steps / max(rounds, 1) for rounds in result.rounds_per_trajectory)
    return {
        "baseline_calls": num_steps,
        "target_calls": result.target_calls,
        "end_to_end_baseline_calls": num_steps + deterministic_steps,
        "end_to_end_target_calls": result.target_calls + deterministic_steps,
        "isolated_speedup_sum": isolated,
        "sample_count": result.batch_size,
        "occupancy_active": active,
        "occupancy_slots": slots,
        "accepted_levels": accepted,
        "verified_levels": verified,
        "target_states_evaluated": result.target_states_evaluated,
    }


def add_metrics(total: dict, part: Mapping[str, Any]) -> None:
    for key in _METRIC_KEYS:
        total[key] = total.get(key, 0) + part[key]


def summarise_metrics(total: Mapping[str, Any]) -> dict:
    def ratio(numerator, denominator):
        return float(numerator) / max(float(denominator), 1.0)

    return {
        "speedup": ratio(total["baseline_calls"], total["target_calls"]),
        "end_to_end_speedup": ratio(
            total["end_to_end_baseline_calls"], total["end_to_end_target_calls"]
        ),
        "mean_isolated_speedup": ratio(
            total["isolated_speedup_sum"], total["sample_count"]
        ),
        "occupancy": ratio(total["occupancy_active"], total["occupancy_slots"]),
        "acceptance_rate": ratio(total["accepted_levels"], total["verified_levels"]),
    }


def validate_reusable_shard(
    path: Path, *, signature: Mapping[str, Any], rank: int, start: int, count: int
):
    """Load a shard only if it is exactly the block this invocation expects."""
    part = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"run_signature": signature, "rank": rank, "start": start, "count": count}
    mismatches = [key for key, value in expected.items() if part.get(key) != value]
    samples = part.get("samples")
    if not isinstance(samples, torch.Tensor) or int(samples.shape[0]) != count:
        mismatches.append("samples")
    if mismatches:
        fields = ", ".join(sorted(set(mismatches)))
        raise SystemExit(
            f"{path}: existing shard is incompatible ({fields}); "
            "use --overwrite or a fresh --out directory"
        )
    return part


def load_shards(
    out: Path,
    *,
    signature: Mapping[str, Any],
    num_samples: int,
    world: Optional[int] = None,
):
    """Load a complete contiguous shard set and reject stale/extra files."""
    shards = sorted(out.glob("shard_*.pt"))
    if world is not None:
        expected_names = [f"shard_{rank:03d}.pt" for rank in range(world)]
        if [p.name for p in shards] != expected_names:
            raise SystemExit(
                f"{out}: expected shards {expected_names}, found {[p.name for p in shards]}"
            )
    if not shards:
        raise SystemExit(f"{out}: no shards to merge")

    parts = [torch.load(path, map_location="cpu", weights_only=True) for path in shards]
    parts.sort(key=lambda part: int(part.get("rank", -1)))
    cursor = 0
    for rank, part in enumerate(parts):
        if part.get("run_signature") != signature:
            raise SystemExit(f"{shards[rank]}: run signature does not match this invocation")
        if part.get("rank") != rank or part.get("start") != cursor:
            raise SystemExit(f"{out}: shards are not contiguous and rank ordered")
        count = int(part.get("count", -1))
        samples = part.get("samples")
        if (count < 1 or not isinstance(samples, torch.Tensor)
                or int(samples.shape[0]) != count):
            raise SystemExit(f"{out}: shard {rank} has inconsistent sample count")
        cursor += count
    if cursor != num_samples:
        raise SystemExit(f"{out}: shards contain {cursor} samples, expected {num_samples}")
    return shards, parts


def merged_metrics(parts) -> tuple[dict, dict]:
    total = {}
    for part in parts:
        metrics = part.get("metric_totals")
        if not isinstance(metrics, dict) or any(key not in metrics for key in _METRIC_KEYS):
            raise SystemExit("shard lacks additive metric counters; regenerate with --overwrite")
        add_metrics(total, metrics)
    return total, summarise_metrics(total)


def save_grid(samples: torch.Tensor, path: Path) -> None:
    """Save up to 64 images without dropping a non-square tail."""
    import PIL.Image

    n = min(64, int(samples.shape[0]))
    if n < 1:
        raise ValueError("cannot make a grid from zero samples")
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    h, w = int(samples.shape[2]), int(samples.shape[3])
    grid = PIL.Image.new("RGB", (cols * w, rows * h))
    for i in range(n):
        array = samples[i].permute(1, 2, 0).numpy()
        grid.paste(PIL.Image.fromarray(array), ((i % cols) * w, (i // cols) * h))
    grid.save(path)
