# Stable Diffusion 3.5 protocol

This protocol generates images from a caption set with SD3.5-medium and measures prompt
alignment. It follows the CIFAR-10 workflow but changes the conditioning, quality metric, and
memory requirements.

## Key differences

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

## Cards below ~16 GiB: `--encode-device cpu`

By default the whole pipeline goes to one GPU, so the load alone needs ~18 GiB.
`--encode-device cpu` runs the text encoders on the CPU instead. Only the
transformer and the VAE ever reach the GPU, which drops the peak to about
4.8 GiB. The text encoders run once, before the first latent exists, and
`free_text_encoders` drops them immediately afterwards — so this changes *where*
the conditioning is computed, and nothing about what is sampled.

A CPU text encoder loads in float32, because bfloat16 has no useful CPU kernels
(T5-XXL measures ~3x slower in it). Encoding 100 captions costs about 2.5
minutes per run. The embeddings are cast back to `--dtype` before the
transformer sees them.

Measured on a 10 GiB RTX 3080, at the heaviest cell of the default grid
(`K=7, L=2`, so `|I|=8`):

| setting | peak | result |
| --- | --- | --- |
| default (no `--encode-device`) | — | fails during the load |
| `--encode-device cpu --decode-batch 8` | 9999 MiB | fails in the VAE decode |
| `--encode-device cpu --decode-batch 2` | 8839 MiB | works |

**Lower `--decode-batch` before `--sample-batch`.** At 512px the VAE decode
peaks higher than the transformer does, so it is the first thing to fail, and
it costs nothing to shrink — the decode is a fixed cost per image either way.

A driver note. On driver 470 an out-of-memory failure is reported as
`RuntimeError: CUDA driver error: invalid argument`, not as the usual
`CUDA out of memory`. Read that message as "too big", and lower
`--decode-batch`.

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

## 1. Run a smoke test

```bash
python experiments/images/run_sd3.py --network stabilityai/stable-diffusion-3.5-medium --no-accelerate --prompts ~/coco30k.txt --rule d-grs --branching 2 --lookahead 3 --num-samples 8 --num-steps 28 --eps 0.25 --guidance-scale 7.0 --sample-batch 2 --forward-batch 16 --device cuda:4 --out /tmp/probe-sd3
```

Expect, in order:

- `512px from (16, 64, 64) latents, T=28 (28 sampler transitions, 2 deterministic)`;
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
file and the index rule. `clip.json` is reused only when the model, dtype,
samples, metadata and prompt source still match. CLIP loads once for every
cell, and text embeddings are cached across cells, so a twelve-cell sweep
encodes each caption once rather than twelve times.

**Use `--baseline`.** Every image's score is kept, not just the mean, and cells
generated from the same (caption, seed) pairs can then be compared *paired*.
The scorer verifies that pairing signature and refuses mismatched seeds or
prompt orderings:

```bash
python experiments/images/clip.py --samples results/sd3/plain-target results/sd3/K2_L3/d-grs results/sd3/K2_L3/rmc --baseline results/sd3/plain-target --device cuda:4 --output results/sd3/clip_report.json
```

The report includes paired and unpaired standard errors. Pairing removes much of the
per-caption variation and provides greater sensitivity when marginal means overlap.

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

## Compatibility note

Checkpoint interfaces can vary across Diffusers versions. Run the smoke test in step 1 to
verify the installed pipeline, prompt encoding, transformer interface, and VAE scaling before
starting a full experiment.


## Endpoint handling and logical NFE accounting

The SD3 sampler starts at pure noise `x_0` and handles all `T` transitions,
including the zero-variance first and final transitions. At a deterministic
step it commits the target mean; continuation requires exact equality with the
drafted state and mean. Initializing the delayed drift still costs one target
call, whose exact root mean is reused during verification. Terminal leaves
`x_T` are not sent to the target, since there is no transition at step `T`.

One logical batched target evaluation counts as one NFE, regardless of
`--forward-batch`, classifier-free guidance, or the number of tree nodes.
`metric_totals` retains rounds and saves `target_calls_per_trajectory`,
`target_states_per_trajectory`, per-image acceptance/committed counts, and
`batch_sizes` / `target_calls_per_batch`. Thus per-image speedups `T / C_i`
and pooled batch speedup `batches * T / sum(C_batch)` can be recomputed later.
Initialization and refinement are included in these direct call counts.
Aggregate proposal, refinement and verification call/row counters and exact
mean reuse counts are also saved.
For SD3, `speedup` and `end_to_end_speedup` now cover the same full trajectory.
The legacy field `speculative_steps` equals `T`; `stochastic_steps` records
only positive-variance transitions.

New SD3 runs carry `endpoint_policy="in_sampler"` and
`nfe_accounting="logical_calls_per_image_v2"`. The sweep rejects completed
cells with the older policy: use a new output root when rerunning comparisons.
Legacy plotting falls back to rounds when direct call counts are absent;
those historical values do not retroactively include initialization.

D-GRS accepts the same residual-complement options as PAWS:

```bash
--verifier-options '{"d-grs":{"residual_complement":"nearest_projection"}}'
```

For `sweep_sd3.sh`, set the same JSON through `VERIFIER_OPTIONS`. Choices are `first` (default),
`fresh`, and `nearest_projection`. Only the perpendicular Gaussian component
on rejection changes; acceptance and the scalar residual law are unchanged.
