#!/usr/bin/env python
"""Paired verification-vs-budget comparison at K=2, L=3, on two GPUs.

Everything except the leaf treatment is held fixed: same prompts, same seeds,
same keyed random streams, `sample_batch=1`, same forward chunk. Sharding is by
image, so each image runs BOTH arms on the SAME GPU -- the contrast can never
pick up a cross-device bf16 difference. Keyed streams are keyed on the global
prompt id, so the sharded union equals a single-GPU run.

The point is to separate three explanations for the plotted inversion:

  (1) randomness/batching artefact  -> killed by construction here
  (2) a bug                         -> the label control must come out exact
  (3) "proximity in state => proximity in drift" fails on a real denoiser
                                    -> measured directly, on identical states

    python experiments/images/sd3_carry_pairing.py --gpus 0,3 --num-prompts 16
    python experiments/images/sd3_carry_pairing.py --backend toy --gpus cpu   # smoke test

Writes CSVs and a printed report to results/sd3-carry-pairing/<timestamp>/.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "specdiff/batched.py").exists())
sys.path.insert(0, str(ROOT))

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
POLICIES = ("parent", "nearest", "nearest-parent", "nearest-all")


def shard_ids(num_prompts: int, num_shards: int, shard: int) -> list:
    """Round-robin, so every shard sees a mix of captions."""
    return list(range(shard, num_prompts, num_shards))


# ------------------------------------------------------------------ worker
def run_worker(args) -> None:
    """One shard. Logs to a file and reports progress through a JSON file, so
    the parent can own the only progress bar on the terminal."""
    import time
    import torch
    from experiments.images.sd3_ablation import run_arm
    from experiments.images.sd3_models import SD3Denoiser

    out = Path(args.out)
    log = (out / f"worker{args.shard}.log").open("w", buffering=1)
    state = {"stage": "load", "done": 0, "total": 0, "detail": ""}

    def note(msg):
        log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def progress(**changes):
        state.update(changes)
        tmp = out / f".progress_shard{args.shard}.tmp"
        tmp.write_text(json.dumps(state))
        tmp.replace(out / f"progress_shard{args.shard}.json")   # atomic for the reader

    progress()
    prompts = [p for p in Path(args.prompts).read_text().splitlines() if p.strip()][: args.num_prompts]
    if len(prompts) < args.num_prompts:
        raise SystemExit(f"{args.prompts} has {len(prompts)} captions, need {args.num_prompts}")
    ids = shard_ids(args.num_prompts, args.num_shards, args.shard)
    if not ids:
        progress(stage="idle")
        return

    # Stage 1: load. Minutes on the real model -- CPU text encoding dominates.
    t0 = time.time()
    note(f"stage 1/3 loading {'toy pipeline' if args.backend == 'toy' else MODEL_ID}"
         f" (text encoders on {args.encode_device})")
    if args.backend == "toy":
        from experiments.images.toy_sd3 import ToySD3Pipeline
        torch.set_num_threads(2)
        denoiser = SD3Denoiser(ToySD3Pipeline(resolution_px=32), prompts,
                               guidance_scale=args.guidance, resolution_px=32)
    else:
        denoiser = SD3Denoiser.from_pretrained(
            MODEL_ID, device="cuda:0",              # CUDA_VISIBLE_DEVICES pins the physical card
            dtype=torch.bfloat16, encode_device=args.encode_device,
            prompt=prompts, guidance_scale=args.guidance, resolution_px=args.resolution,
            # Per-shard cache dir: the key is identical across shards, so one
            # shared dir would race a reader against a writer.
            prompt_cache=str(out.parent / f"_prompt_cache_shard{args.shard}"),
            free_text_encoders=True)
    note(f"stage 1/3 done in {time.time() - t0:.0f}s"
         f"{' (prompt cache hit)' if getattr(denoiser, 'prompt_cache_hit', False) else ''}"
         f"; {len(ids)} images: {ids}")

    common = dict(rule=args.rule, carry="nearest", rng_mode="keyed",
                  sample_batch=1, forward_batch=args.forward_batch, K=2, L=3,
                  num_steps=args.num_steps, eps=args.eps, shift=args.shift,
                  diagnose_drift=True, record_random=False)

    # Stage 2: label-only control. `match` differs but leaves are forced off in
    # both, so for a tree rule these must agree exactly. A mismatch is broken
    # determinism, not a leaf-treatment effect -- stop and investigate.
    t1 = time.time()
    progress(stage="control")
    note("stage 2/3 label control (leaves forced off in both arms)")
    ctl = {m: run_arm(denoiser, image_ids=ids[:2], seed=args.seeds[0], match=m,
                      evaluate_leaves=False, **{**common, "diagnose_drift": False})
           for m in ("verification", "budget")}
    control = dict(
        shard=args.shard,
        same_calls=[r["target_calls"] for r in ctl["verification"].images]
                   == [r["target_calls"] for r in ctl["budget"].images],
        same_latents=bool(ctl["verification"].latents.eq(ctl["budget"].latents).all()),
    )
    ok = all(v for k, v in control.items() if k != "shard")
    note(f"stage 2/3 done in {time.time() - t1:.0f}s: {control}")
    if not ok:
        note("!! label control FAILED -- treatment differences are not interpretable")

    # Stage 3: the grid. One `run_arm` call per image; at sample_batch=1 that is
    # bit-identical to one call over all ids (tape, sampler and init are all
    # built per chunk and keyed on the global prompt id) and it lets progress
    # advance per image instead of per arm.
    images, carries = [], []
    total = len(args.seeds) * 2 * len(ids)
    progress(stage="grid" if ok else "grid(control failed)", total=total)
    note(f"stage 3/3 grid: {len(args.seeds)} seeds x 2 arms x {len(ids)} images = {total} runs")
    n = 0
    for seed in args.seeds:
        for match in ("verification", "budget"):
            arm = []
            for i in ids:
                progress(detail=f"seed{seed} {match[:5]} img{i}")
                res = run_arm(denoiser, image_ids=[i], seed=seed, match=match, **common)
                images += res.images
                carries += res.carries
                arm += res.images
                n += 1
                progress(done=n)
            note(f"seed={seed} {match:<12s} mean NFE speedup "
                 f"{np.mean([r['speedup'] for r in arm]):.4f}")

    pd.DataFrame(images).to_csv(out / f"images_shard{args.shard}.csv", index=False)
    pd.DataFrame(carries).to_csv(out / f"carries_shard{args.shard}.csv", index=False)
    (out / f"control_shard{args.shard}.json").write_text(json.dumps(control, indent=2))
    note(f"all stages done in {time.time() - t0:.0f}s")
    progress(stage="done")


# ---------------------------------------------------------------- analysis
def report(out: Path) -> None:
    from scipy import stats

    images = pd.concat([pd.read_csv(p) for p in sorted(out.glob("images_shard*.csv"))])
    carries = pd.concat([pd.read_csv(p) for p in sorted(out.glob("carries_shard*.csv"))])
    say = lambda *a: print(*a, flush=True)

    say("\n" + "=" * 78 + "\n1. ARM SUMMARY (per-image NFE speedup = num_steps / target_calls)\n")
    say(images.groupby("match").agg(
        images=("image_id", "size"), speedup=("speedup", "mean"),
        target_calls=("target_calls", "mean"), target_rows=("target_states", "mean"),
        rounds=("rounds", "mean"), acceptance=("acceptance_rate", "mean")).round(4).to_string())

    say("\n" + "=" * 78 + "\n2. PAIRED CONTRAST  budget - verification  (same seed, same image)\n")
    wide = images.pivot_table(index=["seed", "image_id"], columns="match", values="speedup")
    d = (wide["budget"] - wide["verification"]).dropna()
    t, p = stats.ttest_rel(wide["budget"], wide["verification"])
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    say(f"   n = {len(d)} paired observations")
    say(f"   mean difference {d.mean():+.4f}   95% CI [{d.mean()-half:+.4f}, {d.mean()+half:+.4f}]   p = {p:.3g}")
    say(f"   budget wins on {100*(d>0).mean():.1f}% of images, ties {100*(d==0).mean():.1f}%")

    say("\n" + "=" * 78 + "\n3. WHAT EACH ARM ACTUALLY CARRIED\n")
    tab = carries.groupby(["match", "reason"]).agg(
        n=("selected_node", "size"), to_parent=("selected_parent", "mean"),
        exact=("exact", "mean"), time_offset=("timestep_offset", "mean"),
        distance=("distance", "mean")).round(4)
    tab["share"] = (tab.n / carries.groupby("match").size()).round(3)
    say(tab.to_string())

    if "error_nearest" not in carries.columns:
        say("\n(no drift diagnostics recorded)")
        return

    # Budget arm only: every candidate exists there, so all four policies are
    # scored on the SAME committed state and the SAME evaluated tree.
    b = carries[carries.match.eq("budget")].copy()
    say("\n" + "=" * 78 + "\n4. DOES PROXIMITY IN STATE IMPLY PROXIMITY IN DRIFT?")
    say("   (budget arm; carry error = ||predicted - true drift|| / sigma)\n")
    b["farther"] = b.parent_distance > b.distance
    b["parent_better"] = b.error_parent < b.error_nearest
    b["misleads"] = b.farther & b.parent_better
    say(b.groupby("reason").agg(
        n=("distance", "size"), dist_carried=("distance", "mean"),
        dist_parent=("parent_distance", "mean"), delta_carried=("delta_nearest", "mean"),
        delta_parent=("delta_parent", "mean"),
        **{"delta_nearest-all": ("delta_nearest-all", "mean")},
        parent_farther=("farther", "mean"), parent_better=("parent_better", "mean"),
        proximity_misleads=("misleads", "mean")).round(4).to_string())

    rej = b[b.reason.ne("full_accept")]
    if len(rej) > 2:
        rho, pr = stats.spearmanr(rej.parent_distance - rej.distance,
                                  rej.error_parent - rej.error_nearest)
        say(f"\n   Spearman( distance gap , drift-error gap ) over {len(rej)} rejections: "
            f"rho = {rho:+.3f}  (p = {pr:.3g})")
        say("   rho near +1 => proximity predicts drift quality; rho near 0 => it carries no information.")

    say("\n" + "=" * 78 + "\n5. COUNTERFACTUAL COST OF NOT EVALUATING LEAVES")
    say("   On the budget arm's own states: what it carried vs the parent the")
    say("   verification arm is forced onto in exactly these situations.\n")
    forced = b[b.reason.isin(["full_accept", "terminal_reject"])]
    if len(forced):
        g = forced.groupby("reason").agg(
            n=("distance", "size"), delta_carried=("delta_nearest", "mean"),
            delta_parent=("delta_parent", "mean")).round(4)
        g["ratio"] = (g.delta_parent / g.delta_carried.replace(0, np.nan)).round(2)
        say(g.to_string())
    say(f"\nSaved to {out}")


# ---------------------------------------------------------------- launcher
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="0,3", help="comma-separated device ids, or 'cpu'")
    ap.add_argument("--backend", default="sd3", choices=("sd3", "toy"))
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rule", default="paws")
    ap.add_argument("--prompts", default=str(ROOT / "experiments/images/coco30k_val2014.txt"))
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--forward-batch", type=int, default=16)
    ap.add_argument("--encode-device", default="cpu")
    ap.add_argument("--out", default=None)
    # internal
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.worker:
        run_worker(args)
        return

    gpus = [g for g in args.gpus.split(",") if g.strip()]
    out = Path(args.out or ROOT / "results/sd3-carry-pairing" /
               datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"out": str(out)}, indent=2, default=str))
    runs = args.num_prompts * len(args.seeds) * 2
    print(f"K=2 L=3  {args.rule}  {args.num_prompts} prompts x {len(args.seeds)} seeds "
          f"x 2 arms = {runs} runs, on {len(gpus)} device(s) {gpus}\n"
          f"writing to {out}\n"
          f"stages per worker: [1/3] load model  [2/3] label control  [3/3] grid\n"
          f"worker logs stream to {out}/worker*.log", flush=True)

    t0 = time.time()
    procs = []
    for i, gpu in enumerate(gpus):
        env = dict(os.environ)
        if gpu != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [sys.executable, __file__, "--worker", "--shard", str(i),
               "--num-shards", str(len(gpus)), "--out", str(out),
               "--backend", args.backend, "--num-prompts", str(args.num_prompts),
               "--seeds", ",".join(map(str, args.seeds)), "--rule", args.rule,
               "--prompts", args.prompts, "--num-steps", str(args.num_steps),
               "--eps", str(args.eps), "--shift", str(args.shift),
               "--guidance", str(args.guidance), "--resolution", str(args.resolution),
               "--forward-batch", str(args.forward_batch),
               "--encode-device", args.encode_device]
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env))
    # One bar, owned by the parent: the workers only write progress files, so
    # nothing else touches the terminal while it is live.
    from tqdm import tqdm
    STAGES = {"load": "loading model", "control": "label control",
              "grid": "grid", "done": "done", "idle": "idle"}

    def poll():
        done, stages, detail = 0, [], ""
        for i in range(len(gpus)):
            try:
                st = json.loads((out / f"progress_shard{i}.json").read_text())
            except (OSError, ValueError):
                continue
            done += int(st.get("done", 0))
            stages.append(STAGES.get(st.get("stage", ""), st.get("stage", "")))
            detail = st.get("detail") or detail
        label = ("starting" if not stages else
                 stages[0] if len(set(stages)) == 1 else "/".join(stages))
        return done, f"[{label}] {detail}".strip()

    bar = tqdm(total=runs, unit="run", dynamic_ncols=True, smoothing=0.05,
               bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    while any(pr.poll() is None for pr in procs):
        time.sleep(1.0)
        done, desc = poll()
        bar.set_description_str(desc)
        bar.n = min(done, runs)
        bar.refresh()
    done, desc = poll()
    bar.n = min(done, runs)
    bar.set_description_str(desc)
    bar.close()

    codes = [pr.wait() for pr in procs]
    for i in range(len(gpus)):                       # worker logs, now that the bar is gone
        path = out / f"worker{i}.log"
        if path.exists():
            print(f"\n----- worker {i} ({gpus[i]}) -----")
            print(path.read_text().rstrip())
    if any(codes):
        raise SystemExit(f"worker(s) failed with {codes}")
    print(f"\nsampling wall clock: {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / max(runs, 1):.1f}s per run)", flush=True)
    report(out)


if __name__ == "__main__":
    main()
