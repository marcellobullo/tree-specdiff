"""Generate images from a pretrained EDM checkpoint through specdiff.

One (rule, K, L, eps) configuration per invocation. No metrics: this writes
samples and the NFE accounting, and scoring is a separate step, so a large
generation run is paid once and can be measured many ways.

    python experiments/images/run_edm.py \\
        --network /path/edm-cifar10-32x32-uncond-vp.pkl --edm-repo /path/edm \\
        --rule d-grs --branching 2 --lookahead 3 \\
        --num-samples 64 --num-steps 100 --eps 0.25 \\
        --device cuda:0 --out results/edm/cifar10-dgrs

FFHQ is the same command with a different ``--network``: resolution and channel
count are read off the checkpoint, so nothing else changes.

    --network /path/edm-ffhq-64x64-uncond-vp.pkl

With no checkpoint at hand, ``--toy`` swaps in the closed-form denoiser of
``toy.py`` and runs on CPU. It exercises every line of the pipeline except the
network itself, which is the point: the wiring is what breaks.

Output (``--out``):
    samples.pt   uint8 (N, 3, H, W) -- the layout the FID scorer expects
    meta.json    protocol + NFE accounting
    grid.png     8x8 montage for eyeballing

Conditional and unconditional checkpoints are interchangeable: ``--labels auto``
(the default) reads the checkpoint and either samples unconditionally or draws
one class per image, so swapping ``--network`` between a ``-cond-`` and a
``-uncond-`` file needs no other change.

EDM's conditional networks have no null class, so a ``-cond-`` checkpoint cannot
be run without labels -- ``--labels none`` on one is refused rather than
silently sampling the wrong distribution. See ``models.py``'s module docstring.
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
)

from images import models  # noqa: E402

REPORT_EVERY_S = 60.0     # progress line / progress.json cadence


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--network", help="pretrained EDM .pkl")
    p.add_argument("--edm-repo", help="checkout of NVlabs/edm, required for a .pkl")
    p.add_argument("--toy", action="store_true",
                   help="closed-form stand-in denoiser; no checkpoint, runs on CPU")
    p.add_argument("--toy-resolution", type=int, default=16)
    p.add_argument("--toy-classes", type=int, default=0,
                   help="toy only: label_dim, so the conditional path is "
                        "exercisable without a real checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sample-batch", type=int, default=0,
                   help="trajectories per batched run; 0 = all at once")
    p.add_argument("--num-steps", type=int, default=100, help="the horizon T")
    p.add_argument("--eps", type=float, default=0.25, help="churn")
    p.add_argument("--rule", default="d-grs", choices=("d-grs", "rmc", "target"))
    p.add_argument("--branching", type=int, default=2, help="K")
    p.add_argument("--lookahead", type=int, default=3, help="L")
    p.add_argument("--match", default="verification",
                   choices=("verification", "budget"),
                   help="how an rmc chain is sized against the (K, L) tree it "
                        "is compared with. 'verification' (default, and "
                        "gm_sweep.py's) gives both arms the same target batch "
                        "|I| -- the hardware-matched comparison. 'budget' gives "
                        "them the same proposal budget B, the paper's protocol")
    p.add_argument("--forward-batch", type=int, default=0,
                   help="rows per network forward (0 = the whole tree in one "
                        "call); caps activation memory, exact, does not change NFE")
    p.add_argument("--shift", type=float, default=1.0,
                   help="timestep shift of the sigma grid (EDM's scheduler default)")
    p.add_argument("--labels", default="auto",
                   help="class conditioning: 'auto' (one uniform label per image "
                        "on a conditional checkpoint, none on an unconditional "
                        "one), 'uniform', 'none', or a class index for a fixed "
                        "class. Swapping a *-cond-* and a *-uncond-* --network "
                        "needs no other change under 'auto'")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU even when an accelerator is visible. Needed "
                        "to run a multi-process job on a Mac, where accelerate "
                        "would pick MPS and torch has no MPS c10d::barrier")
    p.add_argument("--no-accelerate", action="store_true",
                   help="skip Accelerator() entirely and run one process on "
                        "--device; handy on a laptop with no accelerate config")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate shards that already exist instead of "
                        "reusing them (the default makes a crashed run resumable)")
    p.add_argument("--check-contract", action="store_true")
    return p.parse_args(argv)


def resolve_label_mode(args, denoiser) -> str:
    """`--labels auto` resolved against what the checkpoint actually is.

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
    """Every image's class, drawn once for the whole run, or ``None``.

    Deliberately drawn for **all** ``num_samples`` up front from a generator
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
    if not args.edm_repo:
        raise SystemExit("--network needs --edm-repo (a NVlabs/edm checkout)")
    return models.EDMDenoiser.from_pickle(
        args.network, args.edm_repo, device=args.device
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
    depth = tree.budget if match == "budget" else tree.verification_budget()
    return max(1, min(depth, num_steps))


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
    if args.rule == "rmc":
        return DraftTree.chain(matched_chain_depth(uniform, num_steps, args.match))
    return uniform


def build_sampler(setting: models.Setting, tree: DraftTree, args):
    if args.rule == "target":
        return BatchedSpeculativeSampler(
            target=setting.target,
            proposal=IdentityProposal(),
            schedule=setting.schedule, tree=tree, verifier=ResampleVerifier(),
            num_steps=setting.num_steps,
        )
    return BatchedSpeculativeSampler(
        target=setting.target,
        proposal=DelayedDriftProposal(setting.target),
        schedule=setting.schedule, tree=tree,
        verifier=create_verifier(args.rule), num_steps=setting.num_steps,
        check_contract=args.check_contract,
    )


def save_grid(samples: torch.Tensor, path: Path) -> None:
    import PIL.Image

    n = min(64, samples.shape[0])
    side = max(1, int(n**0.5))
    h, w = samples.shape[2], samples.shape[3]
    grid = PIL.Image.new("RGB", (side * w, side * h))
    for i in range(side * side):
        arr = samples[i].permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:
            arr = arr.repeat(3, axis=2)
        grid.paste(PIL.Image.fromarray(arr), ((i % side) * w, (i // side) * h))
    grid.save(path)


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


def generate_shard(args, setting, sampler, denoiser, labels, start, count, out, rank):
    """Generate this process's block and write it atomically."""
    shard = out / f"shard_{rank:03d}.pt"
    if shard.exists() and not args.overwrite:
        print(f"rank {rank}: reusing {shard.name}")
        return shard

    batch = args.sample_batch or count
    # Rank-offset, or every process would draw identical starting noise.
    generator = torch.Generator(device=args.device).manual_seed(args.seed + rank)
    n_endpoints = len(setting.deterministic_steps)
    chunks, calls, rows = [], 0, 0
    speedups, end_to_end, isolated, occupancies, accepts = [], [], [], [], []
    t0, done, last_report = time.time(), 0, 0.0

    while done < count:
        n = min(batch, count - done)
        first = start + done                          # global index of this batch
        setting.target.set_class_labels(
            None if labels is None else labels[first : first + n].to(args.device)
        )
        y0 = torch.randn(
            (n, *setting.state_shape), generator=generator,
            device=args.device, dtype=torch.float32,
        )
        y, result = models.sample_trajectory(
            setting, sampler, y0, rng=generator, generator=generator
        )
        chunks.append(models.to_uint8(y).cpu())
        calls += result.target_calls
        rows += result.target_states_evaluated
        speedups.append(result.speedup)
        # The two Euler endpoints are one network call each for the whole batch,
        # and every sampler pays them -- so they belong in the end-to-end figure
        # and not in the speculative-window one.
        end_to_end.append(
            (setting.num_steps + n_endpoints) / (result.target_calls + n_endpoints)
        )
        isolated.append(result.mean_isolated_speedup)
        occupancies.append(result.occupancy)
        accepts.append(result.acceptance_rate)
        done += n

        # tqdm is useless under `accelerate launch` with stderr redirected to a
        # log, so emit a pollable progress file instead. *Every* rank writes its
        # own -- the batch finishes when the slowest one does, so a rank stuck on
        # a contended GPU is exactly what you need to be able to see. Only rank 0
        # prints, because four interleaved progress streams in one log are worse
        # than none:  `cat <out>/progress_rank*.json` shows them all.
        now = time.time()
        if now - last_report > REPORT_EVERY_S or done == count:
            rate = done / max(now - t0, 1e-9)
            with open(out / f"progress_rank{rank:03d}.json", "w") as f:
                json.dump({"rule": args.rule, "rank": rank, "done": done,
                           "of": count, "img_per_s": round(rate, 4),
                           "elapsed_s": round(now - t0, 1)}, f)
            if rank == 0:
                print(f"  rank 0: {done}/{count}  {rate:.2f} img/s  "
                      f"speedup {result.speedup:.2f}x  acc {result.acceptance_rate:.3f}",
                      flush=True)
            last_report = now

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    torch.save(
        {
            "samples": torch.cat(chunks),
            "target_calls": calls,
            "target_states_evaluated": rows,
            "speedup": mean(speedups),
            "end_to_end_speedup": mean(end_to_end),
            "mean_isolated_speedup": mean(isolated),
            "occupancy": mean(occupancies),
            "acceptance_rate": mean(accepts),
            "seconds": round(time.time() - t0, 1),
            "rank": rank,
        },
        str(shard) + ".tmp",
    )
    os.replace(str(shard) + ".tmp", shard)
    return shard


def merge_shards(args, setting, tree, denoiser, label_mode, out):
    """Concatenate the per-rank shards into one `samples.pt` plus `meta.json`."""
    shards = sorted(out.glob("shard_*.pt"))
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in shards]
    samples = torch.cat([p["samples"] for p in parts])
    mean = lambda key: sum(p[key] for p in parts) / len(parts)  # noqa: E731

    meta = {
        "rule": args.rule,
        "network": "toy" if args.toy else args.network,
        "num_samples": int(samples.shape[0]),
        "image_shape": list(samples.shape[1:]),
        "total_steps": setting.total_steps,
        "speculative_steps": setting.num_steps,
        "deterministic_steps": list(setting.deterministic_steps),
        "eps": args.eps,
        "shift": args.shift,
        "branching": args.branching if args.rule != "target" else 1,
        "lookahead": args.lookahead if args.rule != "target" else 1,
        # For rmc these are the (K, L) it was *matched against*, not its own
        # topology -- `chain_depth` is what it actually ran.
        "match": args.match if args.rule == "rmc" else None,
        "chain_depth": tree.depth if args.rule == "rmc" else None,
        "proposal_budget": tree.budget,
        "verification_budget": tree.verification_budget(),
        "sample_batch": args.sample_batch or args.num_samples,
        "forward_batch": args.forward_batch,
        "seed": args.seed,
        "num_processes": len(parts),
        "conditional": denoiser.num_classes > 0,
        "num_classes": denoiser.num_classes,
        # Labels are reproducible from (labels, seed) and independent of the
        # process count, so 50k of them do not belong in every meta.json.
        "labels": label_mode,
        # `speedup` is over the speculative window; `end_to_end_speedup` adds the
        # two deterministic Euler endpoints every sampler pays, and is the figure
        # comparable with implementations that run all T steps through one loop.
        # `mean_isolated_speedup` is what each trajectory would manage alone --
        # the gap to `speedup` is the straggler cost of sharing a batch.
        "speedup": mean("speedup"),
        "end_to_end_speedup": mean("end_to_end_speedup"),
        "mean_isolated_speedup": mean("mean_isolated_speedup"),
        "occupancy": mean("occupancy"),
        "acceptance_rate": mean("acceptance_rate"),
        "target_calls": sum(p["target_calls"] for p in parts),
        "target_states_evaluated": sum(p["target_states_evaluated"] for p in parts),
        # Wall clock is the slowest rank, not the sum: they run concurrently.
        "seconds": max(p["seconds"] for p in parts),
        # Per rank, so a straggler is diagnosable after the fact. A spread here
        # means one GPU was contended or slower, and the whole run waited on it.
        "seconds_per_rank": [p["seconds"] for p in sorted(parts, key=lambda q: q["rank"])],
    }

    torch.save(samples, out / "samples.pt.tmp")
    os.replace(out / "samples.pt.tmp", out / "samples.pt")
    for p in shards:
        p.unlink()
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    save_grid(samples, out / "grid.png")
    for f in out.glob("progress_rank*.json"):
        f.unlink()

    secs = meta["seconds_per_rank"]
    if len(secs) > 1:
        slowest, fastest = max(secs), min(secs)
        print(f"per-rank seconds: {secs}")
        if slowest > 1.15 * fastest:
            print(f"  note: slowest rank took {slowest / fastest:.2f}x the fastest "
                  f"({slowest:.0f}s vs {fastest:.0f}s). Wall clock is the slowest "
                  f"rank, so a contended or slower GPU costs the whole run.")
    return meta


def main(argv=None) -> None:
    args = parse_args(argv)
    # Long tree runs allocate and free many differently-sized activation blocks;
    # without this the allocator can fragment itself out of memory even when the
    # total is fine. Harmless when memory is plentiful. Set before any CUDA
    # context exists, and only if the caller has not chosen their own policy.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    accelerator = None
    rank, world = 0, 1
    if not args.no_accelerate:
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=args.cpu)
        rank, world = accelerator.process_index, accelerator.num_processes
        args.device = str(accelerator.device)

    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    barrier(accelerator)

    denoiser = build_denoiser(args)
    setting = models.build(
        denoiser, num_steps=args.num_steps, eps=args.eps, shift=args.shift,
        forward_batch=args.forward_batch,
    )
    tree = build_tree(args, setting.num_steps)
    sampler = build_sampler(setting, tree, args)
    label_mode = resolve_label_mode(args, denoiser)
    labels = all_labels(label_mode, args.num_samples, denoiser, args.seed)
    start, count = shard_bounds(args.num_samples, rank, world)

    if rank == 0:
        print(f"{args.rule}: {args.num_samples} samples over {world} process(es), "
              f"{setting.state_shape} states, T={setting.total_steps} "
              f"({setting.num_steps} speculative + "
              f"{len(setting.deterministic_steps)} Euler), eps={args.eps}")
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
    print(f"rank {rank}: images {start}..{start + count - 1}", flush=True)

    generate_shard(args, setting, sampler, denoiser, labels, start, count, out, rank)

    barrier(accelerator)
    if rank == 0:
        meta = merge_shards(args, setting, tree, denoiser, label_mode, out)
        print(json.dumps(meta, indent=2))

    # Tear the process group down explicitly. Without this NCCL warns at exit
    # about leaked resources -- harmless, but it is the last thing printed after
    # a long run, which makes a clean run look like a failed one.
    if accelerator is not None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
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
