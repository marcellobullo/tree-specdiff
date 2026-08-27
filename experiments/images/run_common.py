"""Shared, dependency-light bookkeeping for the EDM and SD3 experiment drivers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
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
    # Display and placement choices, not protocol: two runs that differ only
    # here produce identical samples, so a shard from one is reusable by the
    # other. "progress" belongs on this list for the same reason "device" does.
    ignored = {"out", "device", "cpu", "no_accelerate", "overwrite",
               "check_contract", "progress"}
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


# --------------------------------------------------------------------- progress
_BAR_WIDTH = 22


def _clock(seconds: float) -> str:
    """``h:mm:ss`` past an hour, ``m:ss`` below it."""
    seconds = int(max(seconds, 0.0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class ProgressReporter:
    """One live progress line for a whole sharded run.

    Every rank writes its own ``progress_rankNNN.json``; rank 0 adds the peers'
    files to its own counters and draws a single bar for the *global* run, so a
    multi-GPU job reports one line rather than one interleaved line per GPU.
    Throughput is summed over the ranks still working, which is what makes the
    ETA the job's ETA rather than one process's -- ranks rarely run at the same
    speed, and the run ends with the slowest.

    Progress is counted in images. Within a batch the sampler's per-round hook
    contributes a fractional image count (``in_flight``), so a bar with only a
    handful of batches to report still moves: a round is one target call, which
    is the finest granularity that exists here.

    Off a TTY -- a redirected log, ``nohup``, a scheduler -- the bar degrades to
    one plain line every ``plain_every_s`` seconds instead of a redrawn bar.
    """

    def __init__(
        self,
        out: Path,
        *,
        rank: int,
        world: int,
        total: int,
        label: str = "",
        mode: str = "auto",
        stream=None,
        plain_every_s: float = 60.0,
        write_every_s: float = 2.0,
        redraw_every_s: float = 0.25,
    ) -> None:
        self.out = Path(out)
        self.rank, self.world, self.total = int(rank), int(world), int(total)
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.plain_every_s = float(plain_every_s)
        self.write_every_s = float(write_every_s)
        self.redraw_every_s = float(redraw_every_s)
        self.display = self._resolve_display(mode)
        self.path = self.out / f"progress_rank{self.rank:03d}.json"

        self.started = time.time()
        self.done = 0.0          # this rank, images; fractional while a batch runs
        self.of = 0              # this rank's share, set on the first update
        self.fields: dict = {}
        self._last_write = 0.0
        self._last_draw = 0.0
        self._drawn = False

    def _resolve_display(self, mode: str) -> str:
        """Only rank 0 draws; every rank still writes its progress file."""
        if mode == "none" or self.rank != 0:
            return "none"
        if mode in ("bar", "plain"):
            return mode
        try:
            return "bar" if self.stream.isatty() else "plain"
        except Exception:                                     # noqa: BLE001
            return "plain"

    # ---------------------------------------------------------------- reporting
    def update(self, done, of=None, *, in_flight: float = 0.0, force: bool = False,
               **fields) -> None:
        """Record ``done`` images finished by this rank and redraw if it is time."""
        self.done = float(done) + float(in_flight)
        if of is not None:
            self.of = int(of)
        self.fields.update(fields)
        now = time.time()
        if force or now - self._last_write >= self.write_every_s:
            self._write(now)
        if self.display == "none":
            return
        every = self.redraw_every_s if self.display == "bar" else self.plain_every_s
        if force or now - self._last_draw >= every:
            self._draw(now)
            self._last_draw = now

    def close(self) -> None:
        """End the bar's line so later output starts on a fresh one."""
        if self.display == "bar" and self._drawn:
            self.stream.write("\n")
            self.stream.flush()
        self._drawn = False

    # ------------------------------------------------------------------ private
    def _write(self, now: float) -> None:
        elapsed = now - self.started
        payload = {
            "rule": self.label, "rank": self.rank,
            "done": round(self.done, 3), "of": self.of,
            "img_per_s": round(self.done / max(elapsed, 1e-9), 4),
            "elapsed_s": round(elapsed, 1),
            "finished": self.of > 0 and self.done >= self.of,
        }
        payload.update({k: v for k, v in self.fields.items()})
        # Atomic, and per-process: rank 0 reads these files while their owners
        # are writing them, and the pid keeps two processes that believe they
        # are the same rank -- a misconfigured launcher -- off one temp file.
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        self._last_write = now
        try:
            with open(temporary, "w") as handle:
                json.dump(payload, handle)
            os.replace(temporary, self.path)
        except OSError:
            # Progress is bookkeeping. A full or racing filesystem must not take
            # down a generation run that has hours of samples behind it.
            pass

    def _global(self) -> tuple[float, float]:
        """``(images done, images per second)`` summed over the run's ranks."""
        done = self.done
        rate = 0.0 if self.of and self.done >= self.of else self.done / max(
            time.time() - self.started, 1e-9
        )
        for path in self.out.glob("progress_rank*.json"):
            if path == self.path:
                continue
            try:
                with open(path) as handle:
                    peer = json.load(handle)
                peer_done = float(peer.get("done", 0.0))
            except (OSError, ValueError, TypeError):
                continue                       # mid-write or truncated; skip a frame
            done += peer_done
            if not peer.get("finished"):       # a finished rank adds no throughput
                rate += float(peer.get("img_per_s", 0.0) or 0.0)
        if rate <= 0.0 and done > 0.0:         # every rank done: report the average
            rate = done / max(time.time() - self.started, 1e-9)
        return done, rate

    def _line(self, now: float) -> str:
        done, rate = self._global()
        fraction = min(done / self.total, 1.0) if self.total > 0 else 0.0
        parts = [self.label] if self.label else []
        if self.display == "bar":
            filled = int(round(fraction * _BAR_WIDTH))
            parts.append("[" + "#" * filled + "." * (_BAR_WIDTH - filled) + "]")
        parts.append(f"{fraction * 100:3.0f}%")
        parts.append(f"{done:.0f}/{self.total} img")
        parts.append(f"{rate:.2f} img/s")
        remaining = self.total - done
        parts.append(
            f"eta {_clock(remaining / rate)}" if rate > 0.0 and remaining > 0 else "eta --"
        )
        parts.append(f"[{_clock(now - self.started)}]")
        if self.world > 1:
            parts.append(f"{self.world} ranks")
        for key, value in self.fields.items():
            parts.append(f"{key} {value:.3g}" if isinstance(value, float) else f"{key} {value}")
        return "  ".join(parts)

    def _draw(self, now: float) -> None:
        line = self._line(now)
        if self.display != "bar":
            print(line, file=self.stream, flush=True)
            return
        width = shutil.get_terminal_size((100, 20)).columns
        # Pad to the previous width so a shrinking line leaves no debris behind.
        self.stream.write("\r" + line[: max(width - 1, 20)].ljust(width - 1))
        self.stream.flush()
        self._drawn = True
