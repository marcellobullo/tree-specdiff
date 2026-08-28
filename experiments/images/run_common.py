"""Shared, dependency-light bookkeeping for the EDM and SD3 experiment drivers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch

SIGNATURE_VERSION = 1
_METRIC_KEYS = (
    "batches",
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
# Per-image records: concatenated, never summed. Chunks run in image order and
# shards merge rank by rank, so entry `i` belongs to image `i` -- row `i` of
# `samples.pt`, the same mapping the labels use.
_SEQUENCE_KEYS = ("rounds_per_trajectory",)


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


# A directory argument is a checkout, and a checkout holds two populations:
# the source, which decides what a run computes, and metadata that churns on
# its own -- `git fetch` rewrites .git/FETCH_HEAD, importing a module writes
# __pycache__. Folding the second kind into the identity expires every
# resumable shard on disk for a reason with no bearing on the samples, so it is
# left out.
_IGNORED_PARTS = frozenset({".git", "__pycache__"})


def _decides_the_run(relative: Path) -> bool:
    return (not _IGNORED_PARTS.intersection(relative.parts)
            and relative.suffix != ".pyc")


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
    files = sorted(item for item in path.rglob("*")
                   if item.is_file() and _decides_the_run(item.relative_to(path)))
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return {"path": str(path.resolve()), "files": len(files),
            "manifest_sha256": digest.hexdigest()}


# ---------------------------------------------------------------- sampler config
# The sampler's own knobs live in a file rather than on the command line. They
# are protocol, not placement: they change the samples, they are easy to get
# wrong in a way nothing downstream would notice, and a finished run has to be
# able to say which ones produced it. Adding one here is the whole change --
# `load_sampler_config` resolves it, `run_signature` picks it up from `args`,
# and `meta.json` records it.
SAMPLER_OPTIONS = ("prefetch", "evaluate_leaves")


def sampler_defaults() -> dict:
    """The library's own defaults, read off the sampler rather than copied.

    A second copy here could disagree with the code silently, which is the
    exact failure this config exists to prevent.
    """
    import inspect

    from specdiff import BatchedSpeculativeSampler

    params = inspect.signature(BatchedSpeculativeSampler.__init__).parameters
    return {name: params[name].default for name in SAMPLER_OPTIONS}


def print_sampler_template() -> None:
    """Write a complete sampler config -- every option, each at its default.

    Generated rather than kept as a checked-in example, so a file produced this
    way lists every option that exists *now* instead of the ones that existed
    when someone last remembered to update a template.
    """
    print(json.dumps(sampler_defaults(), indent=2, sort_keys=True))


def _check_option(name: str, value: Any) -> Any:
    from specdiff.sampler import PREFETCH_MODES

    if name == "prefetch":
        if value not in PREFETCH_MODES:
            raise SystemExit(
                f"sampler config: prefetch must be one of {sorted(PREFETCH_MODES)}; "
                f"got {value!r}"
            )
    elif name == "evaluate_leaves":
        # bool before int: `True` is an int in Python, but `1` is not a bool,
        # and a config that accepted 1 would be accepting a typo.
        if not isinstance(value, bool):
            raise SystemExit(
                f"sampler config: evaluate_leaves must be true or false; got {value!r}"
            )
    return value


def load_sampler_config(path: Optional[str]) -> dict:
    """Resolve the sampler's protocol knobs from a JSON file, defaults filled in.

    JSON because the library declares no dependencies and targets 3.9: PyYAML
    would be a new one and ``tomllib`` arrived in 3.11. Every driver artefact
    here is already JSON.

    The result is always complete, never the file's subset. It goes into the run
    signature, and a signature that omitted a key would let a shard generated
    under one policy be reused by a run under another -- silently, because the
    samples look the same either way.
    """
    config = sampler_defaults()
    if path is None:
        return config
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"--sampler-config: no such file: {file}")
    try:
        raw = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{file}: not valid JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise SystemExit(
            f"{file}: expected a JSON object of sampler options, "
            f"got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - set(SAMPLER_OPTIONS))
    if unknown:
        raise SystemExit(
            f"{file}: unknown sampler option(s): {', '.join(unknown)}. "
            f"Known options are {', '.join(SAMPLER_OPTIONS)}. "
            "A key that is read and then ignored is worse than no config file "
            "at all, so this is an error and not a warning."
        )
    for key, value in raw.items():
        config[key] = _check_option(key, value)
    return config


# ------------------------------------------------------------------ run config
# The file is the run signature's input surface and nothing else. `run_signature`
# below already draws that line with its `ignored` set -- which is exactly "two
# runs differing only here produce identical samples" -- so a parameter's
# `where` and its presence in the signature are the same fact, not two facts
# kept in step by hand. A test pins them together.
# Display and placement choices, not protocol: two runs that differ only here
# produce identical samples, so a shard from one is reusable by the other.
# "progress" belongs on this list for the same reason "device" does. The four
# config-layer names are where settings came from and what to print, not what to
# sample; `sampler_config` and `config` are paths, and what matters is the
# resolved values they produced, which are in `vars(args)` already.
#
# This set and each driver's `where="cli"` parameters are the same fact stated
# twice, and a test holds them together.
IGNORED_IN_SIGNATURE = frozenset({
    "out", "device", "cpu", "no_accelerate", "overwrite", "check_contract",
    "progress", "config", "print_config", "config_provenance",
    "sampler_config", "print_sampler_config",
})

CONFIG_VERSION = 1
SECTIONS = ("model", "schedule", "method", "sampling", "conditioning",
            "execution", "sampler")


class Param(NamedTuple):
    """One knob, described once, for every consumer that needs to know it.

    Feeds argparse construction, help text, config validation, the
    defaults/file/CLI merge, template generation, and `meta.json`. There is no
    second copy of a default anywhere.

    ``where`` is the whole schema. ``"config"`` means the value changes the
    samples, so it belongs in the file *and* in the run signature. ``"cli"``
    means it is placement -- where the run lands, what it prints, which device
    it uses -- so it stays on the command line and out of both.
    """

    name: str                 # argparse dest and config key: "num_steps"
    section: str              # one of SECTIONS; "" when where == "cli"
    default: Any
    kind: type                # bool | int | float | str
    help: str = ""
    choices: Tuple = ()
    check: Optional[Callable[[str, Any], None]] = None
    where: str = "config"
    required: bool = False


def positive(name: str, value: Any) -> None:
    if not value > 0:
        raise SystemExit(f"{name} must be > 0; got {value}")


def non_negative(name: str, value: Any) -> None:
    if value < 0:
        raise SystemExit(f"{name} must be >= 0; got {value}")


def at_least_one(name: str, value: Any) -> None:
    if value < 1:
        raise SystemExit(f"{name} must be >= 1; got {value}")


def check_value(param: Param, value: Any, *, origin: str) -> Any:
    """Type- and range-check one resolved value, whichever layer it came from.

    Runs on the *resolved* record rather than only on the file, so a bad command
    line and a bad config give the same message.
    """
    if value is None:
        if param.default is None:
            return None
        raise SystemExit(f"{origin}: {param.name} may not be null")
    if param.kind is bool:
        # bool before int: `True` is an int in Python, but `1` is not a bool,
        # and a config that accepted 1 would be accepting a typo.
        if not isinstance(value, bool):
            raise SystemExit(
                f"{origin}: {param.name} must be true or false; got {value!r}")
    elif param.kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(
                f"{origin}: {param.name} must be a whole number; got {value!r}")
    elif param.kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(
                f"{origin}: {param.name} must be a number; got {value!r}")
        value = float(value)          # what `type=float` does for `--eps 1`
    elif param.kind is str and not isinstance(value, str):
        raise SystemExit(f"{origin}: {param.name} must be a string; got {value!r}")
    if param.choices and value not in param.choices:
        raise SystemExit(
            f"{origin}: {param.name} must be one of {list(param.choices)}; "
            f"got {value!r}")
    if param.check is not None:
        param.check(param.name, value)
    return value


def defaults(params: Sequence[Param]) -> dict:
    return {p.name: p.default for p in params if p.where == "config"}


def add_arguments(parser: argparse.ArgumentParser, params: Sequence[Param]) -> None:
    """Register every parameter, with no default held by argparse.

    Config parameters get ``argparse.SUPPRESS``, so the parsed namespace carries
    only what the user actually typed -- which is the one thing argparse cannot
    otherwise tell us, and the thing the merge depends on. ``--eps 0.25`` must
    beat a config file even though 0.25 is also the default.
    """
    for p in params:
        flag = "--" + p.name.replace("_", "-")
        kw: dict = {}
        if p.kind is bool:
            if p.where == "config":
                # SUPPRESS would make store_true one-way: a file setting it true
                # could never be overridden back on the command line.
                kw["action"] = argparse.BooleanOptionalAction
            else:
                kw["action"] = "store_true"
        else:
            kw["type"] = p.kind
            if p.choices:
                kw["choices"] = p.choices
        if p.where == "config":
            kw["default"] = argparse.SUPPRESS
            shown = f"(default: {json.dumps(p.default)})"
            kw["help"] = f"{p.help} {shown}" if p.help else shown
        else:
            if kw.get("action") != "store_true":
                kw["default"] = p.default
            if p.help:
                kw["help"] = p.help
        if p.required:
            kw["required"] = True
        parser.add_argument(flag, **kw)


def _sampler_section(body: Any, *, origin: str) -> dict:
    """Validate a `sampler` block against the options the sampler actually has."""
    if not isinstance(body, dict):
        raise SystemExit(
            f'{origin}: section "sampler" must be a JSON object, '
            f"got {type(body).__name__}")
    unknown = sorted(set(body) - set(SAMPLER_OPTIONS))
    if unknown:
        raise SystemExit(
            f"{origin}: unknown sampler option(s): {', '.join(unknown)}. "
            f"Known options are {', '.join(SAMPLER_OPTIONS)}")
    return {k: _check_option(k, v) for k, v in body.items()}


def load_run_config(path: Optional[str], params: Sequence[Param], *,
                    driver: str) -> Tuple[dict, dict]:
    """Read a run config into ``(flat driver values, sampler values)``.

    Both are the file's *subset* -- absent keys stay absent, so the merge can
    tell "the file said so" from "nobody said so". :func:`resolve` is what fills
    in the rest.
    """
    if path is None:
        return {}, {}
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"--config: no such file: {file}")
    try:
        raw = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{file}: not valid JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise SystemExit(
            f"{file}: expected a JSON object with sections "
            f"{', '.join(SECTIONS)}; got {type(raw).__name__}")
    raw = dict(raw)

    version = raw.pop("version", None)
    if version is None:
        raise SystemExit(
            f'{file}: no "version": a run config must declare the schema it was '
            f"written against. The current schema is {CONFIG_VERSION}, and "
            "--print-config emits a complete file")
    if version != CONFIG_VERSION:
        raise SystemExit(
            f"{file}: config schema version {version}, this driver reads "
            f"{CONFIG_VERSION}")
    named = raw.pop("driver", None)
    if named != driver:
        raise SystemExit(
            f'{file}: driver {named!r}, but this is the {driver!r} driver. The '
            "drivers have different parameters and different defaults, so a "
            "config file is written for one of them")

    sampler = _sampler_section(raw.pop("sampler"), origin=str(file)) \
        if "sampler" in raw else {}

    unknown = [s for s in raw if s not in SECTIONS]
    if unknown:
        raise SystemExit(
            f"{file}: unknown section(s): {', '.join(sorted(unknown))}. "
            f"Known sections are {', '.join(SECTIONS)}")

    spec = {p.name: p for p in params}
    homes = {p.name: p.section for p in params if p.where == "config"}
    cli_only = {p.name for p in params if p.where == "cli"}
    values: dict = {}
    for section, body in raw.items():
        if not isinstance(body, dict):
            raise SystemExit(
                f'{file}: section "{section}" must be a JSON object, '
                f"got {type(body).__name__}")
        for key, value in body.items():
            if key in homes and homes[key] != section:
                raise SystemExit(
                    f'{file}: "{key}" is in section "{section}", but it belongs '
                    f'in "{homes[key]}"')
            if key in cli_only:
                raise SystemExit(
                    f'{file}: "{key}" is not a run-configuration option: it is '
                    "placement, not protocol. It does not change the samples, "
                    "so it stays on the command line and out of the run "
                    "signature")
            if key not in homes:
                here = sorted(n for n, sec in homes.items() if sec == section)
                raise SystemExit(
                    f'{file}: unknown option "{key}" in section "{section}". '
                    f"Known options there are {', '.join(here)}. A key that is "
                    "read and then ignored is worse than no config file at all, "
                    "so this is an error and not a warning")
            values[key] = check_value(spec[key], value, origin=str(file))
    return values, sampler


def resolve(params: Sequence[Param], *, file_values: Mapping[str, Any],
            cli_values: Mapping[str, Any]) -> Tuple[dict, dict]:
    """``defaults <- file <- explicit command line``, validated as one record."""
    merged = defaults(params)
    provenance = {name: "default" for name in merged}
    for source, layer in (("file", file_values), ("cli", cli_values)):
        for name, value in layer.items():
            merged[name], provenance[name] = value, source
    spec = {p.name: p for p in params}
    origins = {"default": "default", "file": "config", "cli": "command line"}
    resolved = {
        name: check_value(spec[name], value, origin=origins[provenance[name]])
        for name, value in merged.items()
    }
    return resolved, provenance


def sectioned(flat: Mapping[str, Any], params: Sequence[Param], *, driver: str,
              sampler: Optional[Mapping[str, Any]] = None) -> dict:
    """The flat record, grouped back into the shape the file is written in."""
    homes = {p.name: p.section for p in params if p.where == "config"}
    out: dict = {"version": CONFIG_VERSION, "driver": driver}
    for section in SECTIONS:
        if section == "sampler":
            body = dict(sampler) if sampler else {}
        else:
            body = {n: flat[n] for n in sorted(homes)
                    if homes[n] == section and n in flat}
        if body:
            out[section] = body
    return out


def print_config_template(params: Sequence[Param], *, driver: str) -> None:
    """Write a complete run config -- every option at its default.

    Generated rather than checked in, for the same reason
    :func:`print_sampler_template` is: a file produced this way lists every
    option that exists now, not the ones that existed when a template was last
    remembered.
    """
    print(json.dumps(
        sectioned(defaults(params), params, driver=driver,
                  sampler=sampler_defaults()),
        indent=2, sort_keys=True))


def apply_config(args: argparse.Namespace, params: Sequence[Param], *,
                 driver: str) -> argparse.Namespace:
    """Resolve every parameter onto ``args`` and record where each came from.

    Called at the end of ``parse_args``, so nothing downstream ever sees a
    half-resolved namespace: `run_signature` reads the settings actually in
    force straight out of ``vars(args)``.
    """
    spec = {p.name: p for p in params}
    typed = {k: v for k, v in vars(args).items()
             if k in spec and spec[k].where == "config"}
    file_values, sampler_subset = load_run_config(
        getattr(args, "config", None), params, driver=driver)

    resolved, provenance = resolve(
        params, file_values=file_values, cli_values=typed)
    for name, value in resolved.items():
        setattr(args, name, value)
    args.config_provenance = provenance

    path = getattr(args, "sampler_config", None)
    if path is not None:
        args.sampler = load_sampler_config(path)
    else:
        sampler = sampler_defaults()
        sampler.update(sampler_subset)
        args.sampler = sampler

    # An unresolved parameter would reach the run as whatever argparse left
    # behind -- or as nothing at all -- and land in the signature either way.
    missing = sorted({p.name for p in params} - set(vars(args)))
    if missing:
        raise SystemExit(f"internal: parameters never resolved: {missing}")
    return args


def run_signature(driver: str, args, setting, tree, *, extra: Mapping[str, Any]) -> dict:
    """Versioned signature for deciding whether a shard is safe to resume."""
    config = {k: _plain(v) for k, v in vars(args).items()
              if k not in IGNORED_IN_SIGNATURE}
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
    """Raw counters from one batched sampling call.

    The terms of the ratios :func:`summarise_metrics` reports, which is where
    the speed-ups are defined. A rank sums these over the chunks it runs and
    the merge sums them over the ranks, so every counter here has to be
    additive -- including `batches`, which counts the calls that contributed
    and is what makes `N` and `D` recoverable from the sums afterwards.

    `rounds_per_trajectory` is the exception, concatenated rather than summed:
    `r_i`, the target calls image `i` would have spent on its own. It is the
    only per-image quantity the run records. The two batch ratios reduce a
    batch through a `max` and keep no per-image term at all, so this is what a
    spread across images has to be computed from.
    """
    accepted = sum(sum(r.accepted_depth) for r in result.rounds)
    verified = sum(sum(r.committed) for r in result.rounds)
    active = sum(len(r.active) for r in result.rounds)
    slots = len(result.rounds) * result.batch_size
    isolated = sum(num_steps / max(rounds, 1) for rounds in result.rounds_per_trajectory)
    return {
        "batches": 1,
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
        "rounds_per_trajectory": [int(r) for r in result.rounds_per_trajectory],
    }


def add_metrics(total: dict, part: Mapping[str, Any]) -> None:
    for key in _METRIC_KEYS:
        total[key] = total.get(key, 0) + part[key]
    for key in _SEQUENCE_KEYS:
        total.setdefault(key, []).extend(part[key])


def summarise_metrics(total: Mapping[str, Any]) -> dict:
    r"""The reported speed-ups: ratios of the counters :func:`metric_totals` sums.

    Notation, all of it:

    :math:`T`
        total sampler steps (``--num-steps``).
    :math:`D`
        deterministic endpoint steps: the Euler steps `build` strips, the only
        ones no rule speculates through (``len(deterministic_steps)``).
    :math:`N = T - D`
        speculative steps -- the stretch the rules are compared over, and what
        `num_steps` means everywhere in this module.
    :math:`M`
        images in the run (``--num-samples``), indexed :math:`i = 1 \dots M`.
    :math:`\mathcal{B}`, :math:`|\mathcal{B}|`, :math:`M_b`
        the batches the run splits into and how many there are (the `batches`
        counter): chunks of ``--sample-batch`` rows within a rank, the last one
        short, then over the ranks. Batch :math:`b` holds :math:`M_b` images,
        :math:`\sum_{b \in \mathcal{B}} M_b = M`.
    :math:`r_i`
        rounds trajectory :math:`i` needed, one round being one batched target
        call. Kept per image as `rounds_per_trajectory`.
    :math:`C_b`
        target calls batch :math:`b` spent. One call serves every live row and
        the loop runs until the batch's last trajectory finishes, so

        .. math:: C_b \;=\; \max_{i \in b} r_i .

    The draft tree's own budgets -- :math:`B = K + \dots + K^L` drafted states,
    :math:`|I|` verification rows -- set what a round *costs* and how large a
    batch fits, but they appear nowhere below: a speed-up counts calls, not
    what rides inside one. :math:`B` here is never the batch count.

    The baseline sampler spends one NFE per step: :math:`N` over the
    speculative stretch, :math:`T = N + D` end to end. Hence

    .. math::

        \mathrm{speedup}
            &= \frac{\sum_{b \in \mathcal{B}} N}{\sum_{b \in \mathcal{B}} C_b}
             = \frac{|\mathcal{B}|\,N}
                    {\sum_{b \in \mathcal{B}} \max_{i \in b} r_i} \\[4pt]
        \mathrm{end\_to\_end\_speedup}
            &= \frac{\sum_{b \in \mathcal{B}} (N + D)}
                    {\sum_{b \in \mathcal{B}} (C_b + D)}
             = \frac{|\mathcal{B}|\,(N + D)}
                    {\sum_{b \in \mathcal{B}} C_b \;+\; |\mathcal{B}|\,D} \\[4pt]
        \mathrm{mean\_isolated\_speedup}
            &= \frac{1}{M} \sum_{i=1}^{M} \frac{N}{r_i} .

    Both batch ratios are ratios of sums, not means of ratios: a batch is one
    number, and pooling them pools the terms. Two consequences worth keeping
    straight when reading a table of these:

    .. math::

        \frac{N + D}{C_b + D} \;\le\; \frac{N}{C_b}
            \qquad (C_b \le N: \text{a round commits at least one step}),
        \\[4pt]
        \frac{N}{C_b} \;=\; \frac{N}{\max_{i \in b} r_i}
            \;\le\; \frac{1}{M_b} \sum_{i \in b} \frac{N}{r_i} ,

    so ``end_to_end_speedup`` :math:`\le` ``speedup`` :math:`\le`
    ``mean_isolated_speedup``. The first gap is the two Euler steps nobody
    speculates through; the second is the straggler cost of sharing a batch --
    one call serves every live row, so the batch moves at the pace of its
    slowest member.

    ``occupancy`` and ``acceptance_rate`` are the same shape: pooled ratios of
    the live rows to the slots, and of the accepted levels to the verified
    ones.
    """
    def ratio(numerator, denominator):
        return float(numerator) / max(float(denominator), 1.0)

    summary = {
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
    summary.update(isolated_dispersion(total))
    return summary


def isolated_dispersion(total: Mapping[str, Any]) -> dict:
    r"""Spread of the per-image speed-ups :math:`N / r_i` behind their mean.

    The sample standard deviation and the standard error it implies:

    .. math::

        s = \sqrt{\frac{1}{M - 1} \sum_{i=1}^{M}
                  \Big(\frac{N}{r_i} - \mathrm{mean\_isolated\_speedup}\Big)^2},
        \qquad
        \mathrm{SEM} = \frac{s}{\sqrt{M}} .

    Root-mean-square deviation with Bessel's correction, not the mean absolute
    deviation :math:`\frac{1}{M} \sum_i |N/r_i - \mu|`. Both are honest
    measures of spread; this is the one that divides by :math:`\sqrt{M}` into
    an error bar on the mean, and the one a Gaussian interval assumes.

    Only ``mean_isolated_speedup`` gets a spread here, because it is the only
    reported metric that is a mean over images: ``speedup`` and
    ``end_to_end_speedup`` are one number per run. An error bar on those has to
    come from repeating the run.

    :math:`N` is recovered as ``baseline_calls / batches``, which is why
    `batches` is counted at all.
    """
    rounds = total.get("rounds_per_trajectory")
    batches = int(total.get("batches", 0))
    if not rounds or batches < 1:
        return {}
    num_steps = float(total["baseline_calls"]) / batches
    isolated = [num_steps / max(int(r), 1) for r in rounds]
    std = st.stdev(isolated) if len(isolated) > 1 else 0.0
    return {
        "std_isolated_speedup": std,
        "sem_isolated_speedup": std / len(isolated) ** 0.5,
    }


def signature_difference(theirs: Any, ours: Mapping[str, Any]) -> str:
    """Which settings two run signatures disagree about, in words.

    "incompatible (run_signature)" is true and unhelpful: the signature is a
    record of twenty-odd settings and the answer is nearly always one of them.
    Nested records -- the config, and the checkpoint and checkout identities --
    are opened one level, because "extra differs" is the same non-answer.
    """
    if not isinstance(theirs, Mapping):
        return "run_signature (the shard carries none)"
    named = []
    for key in sorted(set(theirs) | set(ours)):
        mine, yours = theirs.get(key), ours.get(key)
        if mine == yours:
            continue
        if isinstance(mine, Mapping) and isinstance(yours, Mapping):
            named += [
                f"{key}.{sub} (shard {mine.get(sub, '<absent>')!r}, "
                f"this run {yours.get(sub, '<absent>')!r})"
                for sub in sorted(set(mine) | set(yours))
                if mine.get(sub) != yours.get(sub)
            ]
        else:
            named.append(key)
    return "run_signature: " + "; ".join(named) if named else "run_signature"


def validate_reusable_shard(
    path: Path, *, signature: Mapping[str, Any], rank: int, start: int, count: int
):
    """Load a shard only if it is exactly the block this invocation expects."""
    part = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"run_signature": signature, "rank": rank, "start": start, "count": count}
    mismatches = [
        signature_difference(part.get(key), value) if key == "run_signature" else key
        for key, value in expected.items() if part.get(key) != value
    ]
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
            raise SystemExit(
                f"{shards[rank]}: does not match this invocation -- "
                + signature_difference(part.get("run_signature"), signature))
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
        if not isinstance(metrics, dict) or any(
            key not in metrics for key in _METRIC_KEYS + _SEQUENCE_KEYS
        ):
            raise SystemExit("shard lacks additive metric counters; regenerate with --overwrite")
        add_metrics(total, metrics)
    # A short list would be a per-image record that no longer indexes images,
    # which is worse than none: every spread computed from it would be wrong.
    kept = len(total["rounds_per_trajectory"])
    if kept != total["sample_count"]:
        raise SystemExit(
            f"merged metrics carry {kept} per-image NFE counts for "
            f"{total['sample_count']} samples; regenerate with --overwrite"
        )
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
