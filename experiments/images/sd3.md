# Stable Diffusion 3.5 — the full protocol

End-to-end run on SD3.5-medium: generate from a caption set, then score prompt
faithfulness. Structurally like [`cifar10-conditional.md`](cifar10-conditional.md),
but SD3 differs in three ways that change what you run and what you conclude.

## The three differences

**1. Conditioning is text, one caption per image.** `--prompts FILE` gives image
`i` line `i`, and its starting noise comes from seed `--seed + i`. So every rule
sees identical (caption, noise) pairs and the comparison is **paired** — the
arms differ only in the coupling, not in what they were asked to draw. A single
`--prompt` is also supported, but then a whole run is draws from one conditional
and the comparison is far weaker.

**2. There is no FID.** These are samples of a text conditional, not of a
dataset distribution, so there is no real set to be Frechet-distant from. The
quality measurement is **CLIP score** — `max(100·cos(E_image, E_text), 0)` —
which asks whether both rules produce equally prompt-faithful images.

**3. Memory is the binding constraint, not throughput.** SD3.5-medium in bf16 is
~18 GiB of weights before a single latent exists, and classifier-free guidance
**doubles every forward**: a round pushes `sample_batch × |I| × 2` latents
through the transformer. `--forward-batch` stops being optional.

## 0. Prerequisites

```bash
pip install -e '.[all]'
```

Plus a captions file. Build one from COCO 2014 validation:

```bash
python experiments/images/coco_prompts.py --annotations /path/captions_val2014.json --num 30000 --output prompts/coco30k
```

The order depends only on `(--seed, --num-pool)`, never on `--num`, so
`--num 500` is exactly the first 500 lines of the 30k file — take a prefix for a
sweep now and extend to the full set later with nothing to re-derive. It also
writes `.image_ids.txt` and a `.meta.json` recording the source file's SHA-256.
Defaults match the reference implementation, so an existing `coco30k.txt` from
there is reproduced exactly.

## 1. Check it runs before it runs for hours

```bash
python experiments/images/run_sd3.py --network stabilityai/stable-diffusion-3.5-medium --no-accelerate --prompts ~/coco30k.txt --rule d-grs --branching 2 --lookahead 3 --num-samples 8 --num-steps 28 --eps 0.25 --guidance-scale 7.0 --sample-batch 2 --forward-batch 16 --device cuda:4 --out /tmp/probe-sd3
```

Expect, in order:

- `512px from (16, 64, 64) latents, T=28 (26 speculative + 2 Euler)`;
- a `memory:` line showing the CFG doubling, e.g.
  `2 trajectories x |I|=7 = 14 latents per target call, x2 for CFG = 28 rows`;
- `prompts : ~/coco30k.txt (first 8, noise seed i = 0 + i)`;
- `speedup` above 1 and `acceptance_rate` strictly between 0 and 1;
- `grid.png` showing images that plausibly match the first eight captions.

The first run downloads ~16 GiB of weights. Raise `--sample-batch` until it
stops helping or stops fitting; keep `--forward-batch` **fixed** across every
arm you intend to compare.

## 2. Generate the arms

Same `--seed`, `--eps`, `--guidance-scale`, `--forward-batch` and prompts file
everywhere. Baseline first:

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 experiments/images/run_sd3.py --network stabilityai/stable-diffusion-3.5-medium --prompts ~/coco30k.txt --rule target --num-samples 1000 --num-steps 28 --eps 0.25 --guidance-scale 7.0 --seed 0 --sample-batch 16 --forward-batch 16 --out results/sd3/plain-target
```

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 experiments/images/run_sd3.py --network stabilityai/stable-diffusion-3.5-medium --prompts ~/coco30k.txt --rule d-grs --branching 2 --lookahead 3 --num-samples 1000 --num-steps 28 --eps 0.25 --guidance-scale 7.0 --seed 0 --sample-batch 9 --forward-batch 16 --out results/sd3/K2_L3/d-grs
```

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 experiments/images/run_sd3.py --network stabilityai/stable-diffusion-3.5-medium --prompts ~/coco30k.txt --rule rmc --branching 2 --lookahead 3 --num-samples 1000 --num-steps 28 --eps 0.25 --guidance-scale 7.0 --seed 0 --sample-batch 9 --forward-batch 16 --out results/sd3/K2_L3/rmc
```

`rmc` takes the same `(K, L)`; `--match verification` turns it into `chain(7)`,
recorded as `chain_depth` in its `meta.json`.

## 3. Score

```bash
python experiments/images/clip.py --samples results/sd3/plain-target results/sd3/K2_L3/d-grs results/sd3/K2_L3/rmc --device cuda:4 --output results/sd3/clip_report.json
```

Captions are recovered from each run's `meta.json`, which records the prompts
file and the index rule — so the scorer cannot be pointed at the wrong set by
accident, and it refuses rather than guessing. CLIP loads once for every cell,
and text embeddings are cached across cells, so a twelve-cell sweep encodes
each caption once rather than twelve times.

**Use `--baseline`.** Every image's score is kept, not just the mean, and cells
generated from the same (caption, seed) pairs can then be compared *paired*:

```bash
python experiments/images/clip.py --samples results/sd3/plain-target results/sd3/K2_L3/d-grs results/sd3/K2_L3/rmc --baseline results/sd3/plain-target --device cuda:4 --output results/sd3/clip_report.json
```

The paired standard error is far smaller than the unpaired one — the report
prints both — because it sees through the per-caption variation that dominates
the marginal spread. That is what makes a real gap visible at sample counts
where the two means overlap, and it is the payoff of generating every arm from
identical (caption, noise) pairs in the first place.

## 4. What the results must show

**The paired deltas against plain-target must be indistinguishable from zero.**
This is the same check as the FID one for EDM, and the same reason: at
temperature 1 the speculative rules sample *the same law* as plain target
sampling. The report gives a rough `z`; `|z| < 2` is the expected result. A
clearly non-zero paired delta means a coupling bug, not a worse method — and
because it is paired, it will catch a gap the marginal means would hide.

**Speedup above 1** for both speculative arms, `acceptance_rate` strictly
between 0 and 1.

**Look at `grid.png`.** CLIP score is coarse, and a broken latent path can
produce images with the right global statistics and wrong content.

Check `seconds_per_rank` for a contended GPU; wall clock is the slowest rank.

## Sweeping (K, L)

```bash
PROMPTS=~/coco30k.txt GPUS=0,1,2,3 NUM_SAMPLES=1000 EPS=0.25 GUIDANCE=7.0 bash experiments/images/sweep_sd3.sh
```

`NODE_BUDGET` (default 64) sets `--sample-batch = NODE_BUDGET / |I|` per cell,
so rows per forward stay roughly flat as `K` grows — around 104–128 after the
CFG doubling, across the default grid. Start with `NUM_SAMPLES=32
CONFIGS="2,2"` to confirm the launch works before committing.

## What has never been tested

Everything above is exercised against `toy_sd3.py`, a closed-form stand-in
pipeline. **No real SD3.5 weights have been through this code.** Untested:
`StableDiffusion3Pipeline.from_pretrained`, the real `encode_prompt` signature,
the transformer's actual call convention, and the real VAE's scaling factors.
Treat step 1 as the genuine first test.
