"""Generate images from Stable Diffusion 3.5 through specdiff.

This is the latent-space counterpart of ``run_edm.py``. Sharding and accounting
follow the same structure, while conditioning, memory use, and scoring differ.

    accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 \\
        experiments/images/run_sd3.py \\
        --network stabilityai/stable-diffusion-3.5-medium \\
        --prompts prompts/coco30k.txt --guidance-scale 7.0 \\
        --rule d-grs --branching 2 --lookahead 3 \\
        --num-samples 256 --num-steps 28 --eps 0.25 \\
        --sample-batch 4 --forward-batch 16 --out results/sd3/dgrs

Prompts
-------
`--prompt` fixes one caption for the whole run; `--prompts FILE` gives one per
line, and image ``i`` uses line ``i`` with seed ``--seed + i``. This produces a
paired comparison in which every rule receives the same prompt and starting
noise. Results are reproducible independently of GPU count.

Output (`--out`), identical in layout to `run_edm.py` so scoring is shared:
    samples.pt   uint8 (N, 3, H, W) -- decoded pixels, not latents
    meta.json    protocol + NFE accounting
    grid.png     preview montage

Memory
------
Classifier-free guidance **doubles every forward**, so a round asking for
`sample_batch x |I|` latents pushes twice that through the transformer. At
512px, use ``--forward-batch`` to bound activation memory without changing the
algorithm. Freeing the text
encoders after pre-encoding (the default) is what leaves room for a deep tree:
11.2 of the 16.3 GiB an SD3.5-medium pipeline holds is T5-XXL plus the CLIPs.

There is no FID here. These are samples of a *text conditional*, not of a
dataset distribution, so FID against real images is not the measurement; the
NFE and acceptance figures in `meta.json` are, and CLIP score is the quality
side.
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

from images import sd3_models as sd3  # noqa: E402
from images.run_common import (  # noqa: E402
    add_metrics, file_identity, load_shards, merged_metrics, metric_totals,
    run_signature, save_grid, summarise_metrics, validate_reusable_shard,
)

REPORT_EVERY_S = 60.0
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--network", default="stabilityai/stable-diffusion-3.5-medium",
                   help="hub id or a local diffusers directory")
    p.add_argument("--toy", action="store_true",
                   help="closed-form stand-in pipeline; no download, runs on CPU")
    p.add_argument("--toy-resolution", type=int, default=64)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None,
                   help="default: cuda:0 when one is visible, else cpu. Ignored "
                        "under `accelerate launch`, which assigns per rank")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU even when an accelerator is visible")
    p.add_argument("--no-accelerate", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate shards that already exist")
    p.add_argument("--seed", type=int, default=0)
    # what to sample
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--sample-batch", type=int, default=0, help="0 = all at once")
    p.add_argument("--num-steps", type=int, default=28, help="the horizon T")
    p.add_argument("--eps", type=float, default=0.25, help="churn")
    p.add_argument("--rule", default="d-grs", choices=("d-grs", "rmc", "target"))
    p.add_argument("--branching", type=int, default=2, help="K")
    p.add_argument("--lookahead", type=int, default=3, help="L")
    p.add_argument("--match", default="verification", choices=("verification", "budget"),
                   help="how an rmc chain is sized against the (K, L) tree")
    p.add_argument("--shift", type=float, default=sd3.DEFAULT_SHIFT,
                   help="timestep shift; SD3.5 ships 3.0")
    p.add_argument("--forward-batch", type=int, default=0,
                   help="latents per transformer forward (0 = the whole tree in "
                        "one call). Exact -- same RNG, accept/reject and NFE -- "
                        "but not bit-reproducible across values, so keep it "
                        "fixed across a comparison set")
    # conditioning
    p.add_argument("--prompt", default="a photo of a cat",
                   help="one caption for the whole run")
    p.add_argument("--prompts", default=None,
                   help="file of captions, one per line. Image i uses line i at "
                        "seed (--seed + i). Overrides --prompt")
    p.add_argument("--negative-prompt", default="",
                   help="used only when --guidance-scale > 1")
    p.add_argument("--guidance-scale", type=float, default=7.0,
                   help=">1 enables CFG, which doubles every forward")
    p.add_argument("--resolution-px", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES),
                   help="transformer/VAE dtype; the coupling math stays float32")
    p.add_argument("--encode-batch", type=int, default=16,
                   help="captions encoded per text-encoder call")
    p.add_argument("--keep-text-encoders", action="store_true",
                   help="keep the text towers resident after pre-encoding. They "
                        "are 11.2 of 16.3 GiB and never called again")
    p.add_argument("--decode-batch", type=int, default=8,
                   help="latents VAE-decoded per call; the decode peaks higher "
                        "than the transformer at 1024px")
    p.add_argument("--check-contract", action="store_true")
    return p.parse_args(argv)


def load_prompts(path: str, num_samples: int) -> list[str]:
    """Load the first ``num_samples`` lines of a caption file.

    Prefix selection ensures that smaller sweeps and later full runs share
    captions for their common indices.
    """
    lines = Path(path).read_text().splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip()]
    if len(prompts) != len(lines):
        raise SystemExit(f"{path}: blank lines would break the index-to-line map")
    if num_samples > len(prompts):
        raise SystemExit(
            f"--num-samples {num_samples} exceeds the {len(prompts)} captions in "
            f"{path}; each image needs its own"
        )
    return prompts[:num_samples]


def build_denoiser(args) -> sd3.SD3Denoiser:
    prompt = load_prompts(args.prompts, args.num_samples) if args.prompts else args.prompt
    if args.toy:
        from images.toy_sd3 import ToySD3Pipeline

        pipe = ToySD3Pipeline(resolution_px=args.toy_resolution, device=args.device)
        return sd3.SD3Denoiser(
            pipe, prompt, negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale, resolution_px=args.toy_resolution,
            encode_batch=args.encode_batch,
            free_text_encoders=not args.keep_text_encoders,
        )
    return sd3.SD3Denoiser.from_pretrained(
        args.network, device=args.device, dtype=DTYPES[args.dtype],
        prompt=prompt, negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale, resolution_px=args.resolution_px,
        encode_batch=args.encode_batch,
        free_text_encoders=not args.keep_text_encoders,
    )


def matched_chain_depth(tree: DraftTree, num_steps: int, match: str) -> int:
    """Depth of the rmc chain matching `tree`; see `run_edm.matched_chain_depth`."""
    depth = tree.budget if match == "budget" else tree.verification_budget()
    return max(1, min(depth, num_steps))


def build_tree(args, num_steps: int) -> DraftTree:
    if args.rule == "target":
        return DraftTree.chain(1)
    uniform = DraftTree.uniform(branching=args.branching, lookahead=args.lookahead)
    if args.rule == "rmc":
        return DraftTree.chain(matched_chain_depth(uniform, num_steps, args.match))
    return uniform


def build_sampler(setting, tree, args):
    if args.rule == "target":
        return BatchedSpeculativeSampler(
            target=setting.target, proposal=IdentityProposal(),
            schedule=setting.schedule, tree=tree, verifier=ResampleVerifier(),
            num_steps=setting.num_steps,
        )
    return BatchedSpeculativeSampler(
        target=setting.target, proposal=DelayedDriftProposal(setting.target),
        schedule=setting.schedule, tree=tree, verifier=create_verifier(args.rule),
        num_steps=setting.num_steps, check_contract=args.check_contract,
    )


def shard_bounds(num_samples: int, rank: int, world: int):
    """This process's contiguous block `(start, count)` of the global run."""
    per_rank, remainder = divmod(num_samples, world)
    count = per_rank + (1 if rank < remainder else 0)
    start = rank * per_rank + min(rank, remainder)
    return start, count


def barrier(accelerator) -> None:
    if accelerator is None:
        return
    try:
        accelerator.wait_for_everyone()
    except NotImplementedError:                    # no MPS c10d::barrier
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        else:
            raise


def initial_latents(args, setting, denoiser, first: int, n: int, generator):
    """Starting noise for images `first .. first + n - 1`.

    With a prompt set each image's noise comes from its **own** seed, so the
    pair (caption i, noise i) is reproducible from `--seed` alone -- whatever
    the process count or `--sample-batch` was. Every rule then starts from
    identical states, which is what makes the comparison paired rather than two
    marginals. The shared stream is re-seeded per batch so churn and
    accept/reject draws stay deterministic too.
    """
    device = args.device
    if not denoiser.per_sample_prompts:
        return torch.randn((n, *setting.state_shape), generator=generator,
                           device=device, dtype=torch.float32)
    latents = torch.stack([
        torch.randn(setting.state_shape, device=device, dtype=torch.float32,
                    generator=torch.Generator(device=device).manual_seed(args.seed + i))
        for i in range(first, first + n)
    ])
    generator.manual_seed(args.seed + 1_000_003 * first)
    return latents


def decode(denoiser, latents: torch.Tensor, decode_batch: int) -> torch.Tensor:
    """Latents -> uint8 pixels on the CPU, chunked: the VAE peaks above the
    transformer at high resolution."""
    return torch.cat([
        sd3.to_uint8(denoiser.decode_pixels(latents[i : i + decode_batch])).cpu()
        for i in range(0, latents.shape[0], decode_batch)
    ])


def experiment_signature(args, setting, tree, denoiser):
    return run_signature(
        "sd3", args, setting, tree,
        extra={
            "network": file_identity(args.network),
            "prompts": file_identity(args.prompts),
            "per_sample_prompts": denoiser.per_sample_prompts,
        },
    )


def generate_shard(args, setting, sampler, denoiser, start, count, out, rank):
    if count < 1:
        raise SystemExit("each process must receive at least one sample")
    shard = out / f"shard_{rank:03d}.pt"
    signature = experiment_signature(args, setting, sampler.tree, denoiser)
    if shard.exists() and not args.overwrite:
        validate_reusable_shard(
            shard, signature=signature, rank=rank, start=start, count=count
        )
        print(f"rank {rank}: reusing {shard.name}")
        return shard

    batch = args.sample_batch or count
    generator = torch.Generator(device=args.device).manual_seed(args.seed + rank)
    n_endpoints = len(setting.deterministic_steps)
    chunks, metrics = [], {}
    t0, done, last_report = time.time(), 0, 0.0

    while done < count:
        n = min(batch, count - done)
        first = start + done
        denoiser.set_prompt_batch(range(first, first + n))
        y0 = initial_latents(args, setting, denoiser, first, n, generator)
        latents, result = sd3.sample_trajectory(
            setting, sampler, y0, rng=generator, generator=generator
        )
        chunks.append(decode(denoiser, latents, args.decode_batch))
        add_metrics(metrics, metric_totals(
            result, num_steps=setting.num_steps, deterministic_steps=n_endpoints
        ))
        done += n

        now = time.time()
        if now - last_report > REPORT_EVERY_S or done == count:
            rate = done / max(now - t0, 1e-9)
            with open(out / f"progress_rank{rank:03d}.json", "w") as f:
                json.dump({"rule": args.rule, "rank": rank, "done": done, "of": count,
                           "img_per_s": round(rate, 4),
                           "elapsed_s": round(now - t0, 1)}, f)
            if rank == 0:
                print(f"  rank 0: {done}/{count}  {rate:.2f} img/s  "
                      f"speedup {result.speedup:.2f}x  acc {result.acceptance_rate:.3f}",
                      flush=True)
            last_report = now

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


def merge_shards(args, setting, tree, denoiser, out, world=None):
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
        "latent_shape": list(setting.state_shape),
        "total_steps": setting.total_steps,
        "speculative_steps": setting.num_steps,
        "deterministic_steps": list(setting.deterministic_steps),
        "eps": args.eps,
        "shift": args.shift,
        "branching": args.branching if args.rule != "target" else 1,
        "lookahead": args.lookahead if args.rule != "target" else 1,
        "match": args.match if args.rule == "rmc" else None,
        "chain_depth": tree.depth if args.rule == "rmc" else None,
        "proposal_budget": tree.budget,
        "verification_budget": tree.verification_budget(),
        "sample_batch": args.sample_batch or args.num_samples,
        "forward_batch": args.forward_batch,
        "decode_batch": args.decode_batch,
        "seed": args.seed,
        "num_processes": len(parts),
        "prompt": None if args.prompts else args.prompt,
        "prompts_file": args.prompts,
        "prompt_seed_rule": ("image i = line i of prompts_file, noise seed = seed + i"
                             if args.prompts else None),
        "negative_prompt": args.negative_prompt,
        "guidance_scale": args.guidance_scale,
        "resolution_px": args.toy_resolution if args.toy else args.resolution_px,
        "dtype": args.dtype,
        **summary,
        "target_calls": totals["target_calls"],
        "target_states_evaluated": totals["target_states_evaluated"],
        "metric_totals": totals,
        "seconds": max(part["seconds"] for part in parts),
        "seconds_per_rank": [part["seconds"] for part in parts],
        "run_signature": signature,
    }

    torch.save(samples, out / "samples.pt.tmp")
    os.replace(out / "samples.pt.tmp", out / "samples.pt")
    for path in shards:
        path.unlink()
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    save_grid(samples, out / "grid.png")
    for path in out.glob("progress_rank*.json"):
        path.unlink()

    secs = meta["seconds_per_rank"]
    if len(secs) > 1 and min(secs) > 0.0 and max(secs) > 1.15 * min(secs):
        print(f"per-rank seconds: {secs}")
        print(f"  note: slowest rank took {max(secs) / min(secs):.2f}x the fastest; "
              "wall clock is the slowest rank.")
    return meta

def main(argv=None) -> None:
    args = parse_args(argv)
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if args.sample_batch < 0 or args.forward_batch < 0 or args.decode_batch < 1:
        raise SystemExit("batch sizes must be positive, or zero where documented")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    accelerator, rank, world = None, 0, 1
    if not args.no_accelerate:
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=args.cpu)
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
    setting = sd3.build(denoiser, num_steps=args.num_steps, eps=args.eps,
                        shift=args.shift, forward_batch=args.forward_batch)
    tree = build_tree(args, setting.num_steps)
    sampler = build_sampler(setting, tree, args)
    start, count = shard_bounds(args.num_samples, rank, world)

    if rank == 0:
        px = args.toy_resolution if args.toy else args.resolution_px
        print(f"{args.rule}: {args.num_samples} samples over {world} process(es), "
              f"{px}px from {setting.state_shape} latents, T={setting.total_steps} "
              f"({setting.num_steps} speculative + "
              f"{len(setting.deterministic_steps)} Euler), eps={args.eps}")
        print(f"tree {tree}: proposal budget B={tree.budget}, "
              f"verification budget |I|={tree.verification_budget()} rows per round")
        eff = args.sample_batch or args.num_samples
        latents = eff * tree.verification_budget()
        cfg = 2 if args.guidance_scale > 1.0 else 1
        print(f"memory  : {eff} trajectories x |I|={tree.verification_budget()} "
              f"= {latents} latents per target call"
              + (f", x{cfg} for CFG = {latents * cfg} rows" if cfg > 1 else "")
              + (f", split into forwards of {args.forward_batch}"
                 if args.forward_batch > 0 else " (one forward)"))
        print(f"prompts : "
              + (f"{args.prompts} (first {args.num_samples}, noise seed i = "
                 f"{args.seed} + i)" if args.prompts
                 else f'"{args.prompt}" for all {args.num_samples} images'))
    print(f"rank {rank}: images {start}..{start + count - 1}", flush=True)

    generate_shard(args, setting, sampler, denoiser, start, count, out, rank)

    barrier(accelerator)
    if rank == 0:
        print(json.dumps(merge_shards(args, setting, tree, denoiser, out, world=world), indent=2))

    if accelerator is not None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:                                     # noqa: BLE001
        # Hard-exit rather than unwinding: a rank that raises (an OOM while
        # loading the pipeline is the usual one) otherwise blocks in NCCL
        # teardown, and the ranks that did fit wait forever on the barrier for
        # a peer that is never coming.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
