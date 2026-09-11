"""Generate images from a pretrained EDM checkpoint through specdiff.

Each invocation runs one ``(rule, K, L, eps)`` configuration and writes samples
with NFE accounting. Scoring is a separate step, allowing one generation run to
support multiple metrics.

    python experiments/images/run_edm.py \\
        --network /path/edm-cifar10-32x32-uncond-vp.pkl \\
        --rule d-grs --branching 2 --lookahead 3 \\
        --num-samples 64 --num-steps 100 --eps 0.25 \\
        --device cuda:0 --out results/edm/cifar10-dgrs

For FFHQ, change ``--network``; resolution and channel count are read from the
checkpoint.

    --network /path/edm-ffhq-64x64-uncond-vp.pkl

``--toy`` uses the closed-form denoiser in ``toy.py`` and runs on CPU. It
validates the pipeline without loading a checkpoint.

Output (``--out``):
    samples.pt   uint8 (N, 3, H, W) -- the layout the FID scorer expects
    meta.json    protocol + NFE accounting
    grid.png     8x8 preview montage

Progress is one bar for the whole job, sharded or not: every rank writes
``progress_rankNNN.json`` while it works and rank 0 sums them, so a multi-GPU
run reports a single line. ``--progress`` chooses the display; off a terminal it
is one line a minute.

Conditional and unconditional checkpoints are interchangeable: ``--labels auto``
(the default) reads the checkpoint and either samples unconditionally or draws
one class per image, so swapping ``--network`` between a ``-cond-`` and a
``-uncond-`` file needs no other change.

EDM's conditional networks have no null class, so a ``-cond-`` checkpoint cannot
be run without labels. ``--labels none`` is rejected for these checkpoints; see
the ``models.py`` module documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdiff import (  # noqa: E402
    BatchedSpeculativeSampler,
    DelayedDriftProposal,
    DraftTree,
    IdentityProposal,
    ResampleVerifier,
    create_verifier,
    available_verifiers,
)
from specdiff.edm_checkout import default_edm_checkout, is_edm_checkout  # noqa: E402

from images import models  # noqa: E402
from images.run_common import (  # noqa: E402
    Param, ProgressReporter, add_arguments, add_metrics, apply_config,
    at_least_one, build_accelerator, file_identity, load_shards,
    merged_metrics, metric_totals, non_negative, positive,
    print_config_template, print_sampler_template, run_signature, save_grid,
    summarise_metrics, validate_reusable_shard,
)

from experiments.verifier_config import (
    configured_verifier, check_verifier_options, parse_verifier_options,
)

REPORT_EVERY_S = 60.0     # progress-line cadence when there is no bar to redraw


# Every knob this driver has, described once. `where="cli"` is placement --
# where the run lands, what it prints -- and is the same set `run_signature`
# ignores; everything else is protocol, goes in the config file, and is
# recorded. A test pins those two sets together.
PARAMS = (
    Param("network", "model", None, str, help="pretrained EDM .pkl"),
    Param("edm_repo", "model", None, str,
          help="NVlabs/edm checkout (default: <specdiff source>/edm)"),
    Param("toy", "model", False, bool,
          help="closed-form stand-in denoiser; no checkpoint, runs on CPU"),
    Param("toy_resolution", "model", 16, int, check=at_least_one),
    Param("toy_classes", "model", 0, int, check=non_negative,
          help="toy only: label_dim, so the conditional path is "
               "exercisable without a real checkpoint"),

    Param("num_steps", "schedule", 100, int, help="the horizon T",
          check=at_least_one),
    Param("eps", "schedule", 0.25, float, help="churn", check=non_negative),
    Param("s_noise", "schedule", 1.0, float, check=positive,
          help="EDM's S_noise: scales the transition std and not the churn "
               "mean, so it is not a reparameterisation of --eps"),
    Param("shift", "schedule", 1.0, float, check=positive,
          help="timestep shift of the sigma grid (EDM's scheduler default)"),

    Param("rule", "method", "d-grs", str, choices=available_verifiers() + ("target",)),
    Param("verifier_options", "method", "{}", str, check=check_verifier_options,
          help="JSON object of per-rule constructor options, e.g. paws rank_policy"),
    Param("branching", "method", 2, int, help="K", check=at_least_one),
    Param("lookahead", "method", 3, int, help="L", check=at_least_one),
    Param("match", "method", "verification", str,
          choices=("verification", "budget"),
          help="how an rmc chain is sized against the (K, L) tree it "
               "is compared with. 'verification' (default, and "
               "gm_sweep.py's) gives both arms the same target batch "
               "|I| -- the hardware-matched comparison. 'budget' gives "
               "them the same proposal budget B, the paper's protocol"),

    Param("seed", "sampling", 0, int),
    Param("num_samples", "sampling", 64, int, check=at_least_one),
    Param("sample_batch", "sampling", 0, int, check=non_negative,
          help="trajectories per batched run; 0 = all at once"),

    Param("labels", "conditioning", "auto", str,
          help="class conditioning: 'auto' (one uniform label per image "
               "on a conditional checkpoint, none on an unconditional "
               "one), 'uniform', 'none', or a class index for a fixed "
               "class. Swapping a *-cond-* and a *-uncond-* --network "
               "needs no other change under 'auto'"),

    Param("forward_batch", "execution", 0, int, check=non_negative,
          help="rows per network forward (0 = the whole tree in one "
               "call); caps activation memory, exact, does not change NFE"),

    Param("out", "", None, str, where="cli", required=True),
    Param("device", "", "cpu", str, where="cli"),
    Param("cpu", "", False, bool, where="cli",
          help="force CPU even when an accelerator is visible. Needed "
               "to run a multi-process job on a Mac, where accelerate "
               "would pick MPS and torch has no MPS c10d::barrier"),
    Param("no_accelerate", "", False, bool, where="cli",
          help="skip Accelerator() entirely and run one process on "
               "--device; handy on a laptop with no accelerate config"),
    Param("overwrite", "", False, bool, where="cli",
          help="regenerate shards that already exist instead of "
               "reusing them (the default makes a crashed run resumable)"),
    Param("progress", "", "auto", str, where="cli",
          choices=("auto", "bar", "plain", "none"),
          help="rank 0's progress display: 'auto' draws a bar on a "
               "terminal and prints a line every 60s in a log, 'bar' "
               "and 'plain' force one, 'none' silences it. Every rank "
               "writes progress_rankNNN.json either way"),
    Param("check_contract", "", False, bool, where="cli"),
    Param("print_sampler_config", "", False, bool, where="cli",
          help="write a complete sampler config (every option at its "
               "default) to stdout and exit"),
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_arguments(p, PARAMS)
    # The config layer's own controls. They say where settings come from and
    # what to print, not what to sample, so they are neither protocol nor
    # recorded. Mutually exclusive rather than layered: both are grid-level
    # knobs, and no ambiguity beats a precedence rule nobody reads.
    source = p.add_mutually_exclusive_group()
    source.add_argument("--config",
                        help="JSON run configuration; see --print-config. "
                             "Explicit command-line flags override it")
    source.add_argument("--sampler-config",
                        help="JSON file of sampler options alone (prefetch, "
                             "evaluate_leaves) -- the `sampler` section of "
                             "--config, which supersedes this")
    p.add_argument("--print-config", action="store_true",
                   help="write a complete run configuration (every option at "
                        "its default) to stdout and exit")
    args = p.parse_args(argv)
    # Resolved here, not in main(), so `args` carries the settings actually in
    # force: run_signature reads them straight out of vars(args).
    return apply_config(args, PARAMS, driver="edm")


def resolve_label_mode(args, denoiser) -> str:
    """Resolve ``--labels auto`` from checkpoint metadata.

    EDM's conditional networks have no null class -- they were not trained for
    classifier-free guidance -- so a conditional checkpoint cannot be run
    without labels. Sampling the class marginal means drawing one label per
    image, which is what `uniform` does and what the FID protocol expects.
    """
    conditional = denoiser.num_classes > 0
    mode = args.labels
    if mode == "auto":
        return "uniform" if conditional else "none"
    if conditional and mode == "none":
        raise SystemExit(
            f"--labels none on a conditional checkpoint (label_dim = "
            f"{denoiser.num_classes}): EDM has no null class, so this is not "
            "unconditional sampling. Use --labels uniform for the class "
            "marginal, or load a *-uncond-* checkpoint"
        )
    if not conditional and mode != "none":
        raise SystemExit(
            f"--labels {mode} on an unconditional checkpoint (label_dim = 0); "
            "use --labels none or load a *-cond-* checkpoint"
        )
    if mode != "uniform" and not mode.lstrip("-").isdigit():
        raise SystemExit(f"--labels must be auto, uniform, none, or a class index; got {mode!r}")
    return mode


def all_labels(mode: str, num_samples: int, denoiser, seed: int):
    """Return one class per image for the full run, or ``None``.

    Labels are drawn for all ``num_samples`` before sharding from a generator
    seeded only by ``--seed``, then sliced per process -- so image ``i``'s class
    depends on ``i`` and the seed alone, never on how many GPUs the run used or
    what ``--sample-batch`` was. Two runs of the same seed at different process
    counts therefore condition on identical labels, which is what makes their
    FIDs comparable.
    """
    if mode == "none":
        return None
    if mode == "uniform":
        gen = torch.Generator().manual_seed(seed)      # CPU: identical on every rank
        return torch.randint(denoiser.num_classes, (num_samples,), generator=gen)
    return torch.full((num_samples,), int(mode), dtype=torch.long)


def build_denoiser(args) -> models.EDMDenoiser:
    if args.toy:
        from images.toy import GaussianEDMPrecond

        return models.EDMDenoiser(
            GaussianEDMPrecond(
                img_resolution=args.toy_resolution, label_dim=args.toy_classes
            ),
            device=args.device,
        )
    if not args.network:
        raise SystemExit("--network is required (or --toy)")
    edm_repo = (
        Path(args.edm_repo).expanduser()
        if args.edm_repo
        else default_edm_checkout()
    )
    if not is_edm_checkout(edm_repo):
        raise SystemExit(
            f"EDM checkout not found at {edm_repo}. Run specdiff-download-edm "
            "or pass --edm-repo."
        )
    return models.EDMDenoiser.from_pickle(
        args.network, str(edm_repo), device=args.device
    )


def matched_chain_depth(tree: DraftTree, num_steps: int, match: str) -> int:
    """Depth of the RMC chain that matches ``tree`` under the chosen protocol.

    ``verification``
        Equal **target** batch. A chain of depth ``m`` evaluates ``m`` nodes, so
        the matched depth is the tree's ``|I|``. This is the hardware-matched
        arm -- ``|I|`` is what the expensive network actually evaluates and what
        has to fit in memory -- and it is ``gm_sweep.py``'s default.
    ``budget``
        Equal **proposal** budget: a chain of ``B`` drafted states, the paper's
        protocol.

    Either way the chain is clamped to the horizon: a round truncates to
    ``min(depth, N - n)``, so depth beyond ``N`` is unreachable and building it
    would only cost memory. That clamp is also why the two protocols converge --
    once the matched depth exceeds ``N`` both give ``chain(N)``.
    """
    return create_verifier("rmc").matched_tree(
        tree, num_steps=num_steps, match=match
    ).depth


def build_setting(args, denoiser) -> models.Setting:
    """The one place the schedule is assembled from the parsed arguments.

    Extracted so a schedule parameter cannot be threaded here and silently
    defaulted in the tests that rebuild the same object.
    """
    return models.build(
        denoiser, num_steps=args.num_steps, eps=args.eps, shift=args.shift,
        s_noise=args.s_noise, forward_batch=args.forward_batch,
    )


def build_tree(args, num_steps: int) -> DraftTree:
    """The topology this cell runs.

    ``rmc`` is a single-proposal coupling, so it has no tree of its own: given
    ``(K, L)`` it runs the chain matched to that tree under ``--match``. Passing
    the same ``(K, L)`` to both rules is therefore what a sweep does, and the
    matching -- not the caller -- decides how long the chain is.
    """
    if args.rule == "target":
        return DraftTree.chain(1)
    uniform = DraftTree.uniform(branching=args.branching, lookahead=args.lookahead)
    return create_verifier(args.rule).matched_tree(
        uniform, num_steps=num_steps, match=args.match,
        evaluate_leaves=getattr(args, "sampler", {}).get("evaluate_leaves", False)
    )


def build_sampler(setting: models.Setting, tree: DraftTree, args):
    if args.rule == "target":
        return BatchedSpeculativeSampler(
            target=setting.target,
            proposal=IdentityProposal(),
            schedule=setting.schedule, tree=tree, verifier=ResampleVerifier(),
            num_steps=setting.num_steps, **args.sampler,
        )
    return BatchedSpeculativeSampler(
        target=setting.target,
        proposal=DelayedDriftProposal(setting.target),
        schedule=setting.schedule, tree=tree,
        verifier=configured_verifier(
            args.rule, getattr(args, "verifier_options", None)), num_steps=setting.num_steps,
        check_contract=args.check_contract, **args.sampler,
    )



def barrier(accelerator) -> None:
    """Wait for every process, tolerating a broken accelerator barrier.

    ``Accelerator.wait_for_everyone`` picks ``device_ids`` from the current
    accelerator, which on a Mac is MPS -- and torch has no ``c10d::barrier``
    for MPS, so a multi-process CPU job dies there even though the group is a
    perfectly healthy gloo group. Plain ``dist.barrier()`` works fine, so fall
    back to it rather than making the whole sharding path untestable off a
    CUDA box. On CUDA the first call succeeds and the fallback never runs.
    """
    if accelerator is None:
        return
    try:
        accelerator.wait_for_everyone()
    except NotImplementedError:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        else:
            raise


def shard_bounds(num_samples: int, rank: int, world: int):
    """This process's contiguous block ``(start, count)`` of the global run.

    Contiguous and in rank order, so concatenating the shards reproduces global
    order and image ``i`` is the ``i``-th row of ``samples.pt`` -- the mapping
    the label set relies on. The first ``remainder`` ranks take one extra image.
    """
    per_rank, remainder = divmod(num_samples, world)
    count = per_rank + (1 if rank < remainder else 0)
    start = rank * per_rank + min(rank, remainder)
    return start, count


def experiment_signature(args, setting, tree, denoiser):
    return run_signature(
        "edm", args, setting, tree,
        extra={
            "network": "toy" if args.toy else file_identity(args.network),
            "edm_repo": None if args.toy else file_identity(args.edm_repo),
            "num_classes": denoiser.num_classes,
        },
    )


def generate_shard(args, setting, sampler, denoiser, labels, start, count, out,
                   rank, reporter):
    """Generate this process's block and write it atomically."""
    if count < 1:
        raise SystemExit("each process must receive at least one sample")
    shard = out / f"shard_{rank:03d}.pt"
    signature = experiment_signature(args, setting, sampler.tree, denoiser)
    if shard.exists() and not args.overwrite:
        validate_reusable_shard(
            shard, signature=signature, rank=rank, start=start, count=count
        )
        # Report the reused block as finished, so the global bar counts it and
        # this rank contributes no throughput to the ETA -- it has no work left.
        reporter.update(count, count, force=True)
        print(f"rank {rank}: reusing {shard.name}")
        return shard

    batch = args.sample_batch or count
    generator = torch.Generator(device=args.device).manual_seed(args.seed + rank)
    n_endpoints = len(setting.deterministic_steps)
    chunks, metrics = [], {}
    t0, done = time.time(), 0
    reporter.update(0, count)

    while done < count:
        n = min(batch, count - done)
        first = start + done
        setting.target.set_class_labels(
            None if labels is None else labels[first : first + n].to(args.device)
        )
        y0 = torch.randn(
            (n, *setting.state_shape), generator=generator,
            device=args.device, dtype=torch.float32,
        )

        def on_round(steps_taken, steps_total, _done=done, _n=n):
            # Partial credit for the batch in flight: a run with few batches
            # would otherwise sit still for minutes between updates.
            reporter.update(_done, count, in_flight=_n * steps_taken / max(steps_total, 1))

        y, result = models.sample_trajectory(
            setting, sampler, y0, rng=generator, generator=generator,
            on_round=on_round,
        )
        chunks.append(models.to_uint8(y).cpu())
        add_metrics(metrics, metric_totals(
            result, num_steps=setting.num_steps, deterministic_steps=n_endpoints
        ))
        done += n
        reporter.update(
            done, count, force=(done == count),
            spd=round(result.speedup, 2), acc=round(result.acceptance_rate, 3),
        )

    summary = summarise_metrics(metrics)
    torch.save(
        {
            "samples": torch.cat(chunks),
            "rank": rank,
            "start": start,
            "count": count,
            "run_signature": signature,
            "metric_totals": metrics,
            "target_calls": metrics["target_calls"],
            "target_states_evaluated": metrics["target_states_evaluated"],
            **summary,
            "seconds": time.time() - t0,
        },
        str(shard) + ".tmp",
    )
    os.replace(str(shard) + ".tmp", shard)
    return shard


def merge_shards(args, setting, tree, denoiser, label_mode, out, world=None):
    """Validate and concatenate per-rank shards into one completed run."""
    signature = experiment_signature(args, setting, tree, denoiser)
    shards, parts = load_shards(
        out, signature=signature, num_samples=args.num_samples, world=world
    )
    samples = torch.cat([part["samples"] for part in parts])
    totals, summary = merged_metrics(parts)

    meta = {
        "rule": args.rule,
        "network": "toy" if args.toy else args.network,
        "num_samples": int(samples.shape[0]),
        "image_shape": list(samples.shape[1:]),
        "total_steps": setting.total_steps,
        "speculative_steps": setting.num_steps,
        "deterministic_steps": list(setting.deterministic_steps),
        "eps": args.eps,
        "s_noise": args.s_noise,
        "shift": args.shift,
        "branching": args.branching if args.rule != "target" else 1,
        "lookahead": args.lookahead if args.rule != "target" else 1,
        "match": args.match if args.rule != "target" and create_verifier(args.rule).requires_chain else None,
        "chain_depth": tree.depth if args.rule != "target" and create_verifier(args.rule).requires_chain else None,
        "requires_chain": args.rule != "target" and create_verifier(args.rule).requires_chain,
        "proposal_budget": tree.budget,
        "verification_budget": tree.verification_budget(
            evaluate_leaves=args.sampler["evaluate_leaves"]),
        "sample_batch": args.sample_batch or args.num_samples,
        "forward_batch": args.forward_batch,
        "seed": args.seed,
        "num_processes": len(parts),
        "conditional": denoiser.num_classes > 0,
        "num_classes": denoiser.num_classes,
        "labels": label_mode,
        **summary,
        "target_calls": totals["target_calls"],
        "target_states_evaluated": totals["target_states_evaluated"],
        "metric_totals": totals,
        "seconds": max(part["seconds"] for part in parts),
        "seconds_per_rank": [part["seconds"] for part in parts],
        "sampler": dict(args.sampler),
        "verifier_options": parse_verifier_options(args.verifier_options),
        "run_signature": signature,
    }

    torch.save(samples, out / "samples.pt.tmp")
    os.replace(out / "samples.pt.tmp", out / "samples.pt")
    for path in shards:
        path.unlink()
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    save_grid(samples, out / "grid.png")
    for path in out.glob("progress_rank*"):        # including any stale .tmp
        path.unlink()

    secs = meta["seconds_per_rank"]
    if len(secs) > 1:
        slowest, fastest = max(secs), min(secs)
        print(f"per-rank seconds: {secs}")
        if fastest > 0.0 and slowest > 1.15 * fastest:
            print(f"  note: slowest rank took {slowest / fastest:.2f}x the fastest "
                  f"({slowest:.0f}s vs {fastest:.0f}s). Wall clock is the slowest "
                  f"rank, so a contended or slower GPU costs the whole run.")
    return meta

def main(argv=None) -> None:
    args = parse_args(argv)
    if args.print_sampler_config:
        print_sampler_template()
        return
    if args.print_config:
        print_config_template(PARAMS, driver="edm")
        return
    # Long tree runs allocate and free many differently-sized activation blocks;
    # without this the allocator can fragment itself out of memory even when the
    # total is fine. Harmless when memory is plentiful. Set before any CUDA
    # context exists, and only if the caller has not chosen their own policy.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    accelerator = None
    rank, world = 0, 1
    if not args.no_accelerate:
        accelerator = build_accelerator(cpu=args.cpu)
        rank, world = accelerator.process_index, accelerator.num_processes
        args.device = str(accelerator.device)

    if args.num_samples < world:
        raise SystemExit(
            f"--num-samples ({args.num_samples}) must be >= process count ({world})"
        )

    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    barrier(accelerator)

    denoiser = build_denoiser(args)
    setting = build_setting(args, denoiser)
    tree = build_tree(args, setting.num_steps)
    sampler = build_sampler(setting, tree, args)
    label_mode = resolve_label_mode(args, denoiser)
    labels = all_labels(label_mode, args.num_samples, denoiser, args.seed)
    start, count = shard_bounds(args.num_samples, rank, world)

    if rank == 0:
        print(f"{args.rule}: {args.num_samples} samples over {world} process(es), "
              f"{setting.state_shape} states, T={setting.total_steps} "
              f"({setting.num_steps} speculative + "
              f"{len(setting.deterministic_steps)} Euler), eps={args.eps}"
              + (f", s_noise={args.s_noise}" if args.s_noise != 1.0 else ""))
        print(f"tree {tree}: proposal budget B={tree.budget}, "
              f"verification budget |I|={tree.verification_budget()} rows per round")
        # The number that actually sets peak memory, and the one to lower when a
        # run OOMs: one target call carries sample_batch x |I| states.
        eff = args.sample_batch or args.num_samples
        rows = eff * tree.verification_budget()
        print(f"memory  : {eff} trajectories x |I|={tree.verification_budget()} "
              f"= {rows} states per target call"
              + (f", split into forwards of {args.forward_batch}"
                 if args.forward_batch > 0 else " (one forward)"))
        print(f"labels: {label_mode}"
              + (f" over {denoiser.num_classes} classes" if label_mode != "none" else ""))
        # The knobs no other line would reveal, named in the log so a run is
        # identifiable from its output and not only from its meta.json.
        print("sampler : "
              + ", ".join(f"{k}={v}" for k, v in sorted(args.sampler.items()))
              + (f"  (from {args.sampler_config})" if args.sampler_config
                 else "  (defaults)"))
        if args.config:
            # Which settings the file actually supplied, and which the command
            # line took back. Without this, a --config meeting a script that
            # passes flags explicitly looks like a config that was ignored.
            overridden = sorted(k for k, src in args.config_provenance.items()
                                if src == "cli")
            from_file = sum(1 for src in args.config_provenance.values()
                            if src == "file")
            print(f"config  : {args.config} ({from_file} keys)"
                  + (", overridden on the command line: "
                     + ", ".join(overridden) if overridden else ""))
    print(f"rank {rank}: images {start}..{start + count - 1}", flush=True)

    reporter = ProgressReporter(
        out, rank=rank, world=world, total=args.num_samples, label=args.rule,
        mode=args.progress, plain_every_s=REPORT_EVERY_S,
    )
    try:
        generate_shard(args, setting, sampler, denoiser, labels, start, count,
                       out, rank, reporter)
    finally:
        reporter.close()

    barrier(accelerator)
    if rank == 0:
        meta = merge_shards(args, setting, tree, denoiser, label_mode, out, world=world)
        print(json.dumps(meta, indent=2))

    # Tear the process group down explicitly. Without this NCCL warns at exit
    # about leaked resources -- harmless, but it is the last thing printed after
    # a long run, which makes a clean run look like a failed one.
    if accelerator is not None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    # Handled here rather than inside main(): asking what the options are is not
    # asking to run anything, so it must not require --out, and it must not be
    # swallowed by the hard-exit guard below.
    if "--print-sampler-config" in sys.argv[1:]:
        print_sampler_template()
        raise SystemExit(0)
    if "--print-config" in sys.argv[1:]:
        print_config_template(PARAMS, driver="edm")
        raise SystemExit(0)
    try:
        main()
    except BaseException:                                     # noqa: BLE001
        # Hard-exit instead of unwinding. A rank that raises -- an OOM in
        # build_denoiser is the usual one, when a neighbouring job has taken the
        # GPU -- otherwise blocks in NCCL teardown without ever exiting: the
        # launcher sees no dead child and leaves the group up, while the ranks
        # that DID fit generate their whole shard and then wait forever on the
        # post-generation barrier for a peer that is never coming. All the
        # processes sit there holding their memory. Exiting non-zero here is what
        # lets the launcher notice and tear the whole group down.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
