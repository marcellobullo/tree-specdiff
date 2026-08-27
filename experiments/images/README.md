# Image experiments

These experiments integrate Karras et al. (2022) CIFAR-10 and FFHQ checkpoints with
`TargetTransition`, then extend the same workflow to multi-GPU sweeps and SD3.5.

Install the image dependencies and download the tested NVlabs/edm revision into
the repository's `edm/` directory:

```bash
pip install -e '.[edm]' && specdiff-download-edm
# Or use .[all] to include every optional dependency:
pip install -e '.[all]' && specdiff-download-edm
```

`run_edm.py` uses that checkout by default. Pass `--edm-repo` only to use a
different location.

**Pixel space (EDM)**

| file | purpose |
| --- | --- |
| `models.py` | the adapter: denoiser → velocity → churn transition, plus the schedule |
| `toy.py` | a closed-form stand-in denoiser, so the wiring is testable with no checkpoint and no GPU |
| `run_edm.py` | generation driver — writes `samples.pt`, `meta.json`, `grid.png` |
| `sweep.sh` | the `(K, L)` sweep — d-grs vs rmc across the grid, resumable |
| `fid.py` | FID and Inception Score from saved samples, with a cached real set |
| [`cifar10-conditional.md`](cifar10-conditional.md) | full protocol: generate + score, class-conditional CIFAR-10 |
| [`ffhq.md`](ffhq.md) | the same for FFHQ 64x64 — what differs, and why |
| [`server-checklist.md`](server-checklist.md) | smoke tests to pass before either protocol |
| `crosscheck_reference.py` | port fidelity against the sibling implementation |
| `../../tests/test_edm_images.py` | CPU tests for the EDM adapter and experiment utilities |

**Latent space (SD3.5)**

| file | purpose |
| --- | --- |
| `sd3_models.py` | the latent adapter: guided velocity, prompt table, churn transition |
| `toy_sd3.py` | a closed-form stand-in pipeline, so this is testable with no 16 GiB download |
| `run_sd3.py` | generation driver — prompt sets, CFG, chunked VAE decode, sharding |
| `sweep_sd3.sh` | the `(K, L)` sweep for SD3 |
| `clip.py` | CLIP score from saved samples — per-image, so cells can be compared *paired* |
| `coco_prompts.py` | build a deterministic, prefix-stable COCO caption set |
| [`sd3.md`](sd3.md) | full protocol: generate + score |
| `../../tests/test_sd3.py` | CPU tests for the SD3 adapter and experiment utilities |

`sigma_grid`, `churn_std_grid`, and the shard and matching helpers are duplicated between the
pixel and latent implementations to keep each experiment self-contained. `tests/test_sd3.py`
checks that both copies remain equivalent.

SD3 conditions on **text**, one caption per image. `--prompts FILE` gives image
`i` line `i` at noise seed `--seed + i`, so every rule sees identical
(caption, starting noise) pairs and the comparison is paired rather than two
marginals. There is no FID: these are samples of a text conditional, not of a
dataset distribution.

## Run it

```bash
python experiments/images/run_edm.py --toy --rule d-grs --branching 2 --lookahead 3 --num-samples 16 --num-steps 40 --out results/toy
```

The toy configuration requires no checkpoint or GPU. For a pretrained checkpoint:

```bash
python experiments/images/run_edm.py --network /path/edm-cifar10-32x32-uncond-vp.pkl --rule d-grs --branching 2 --lookahead 3 --num-samples 64 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/cifar10-dgrs
```

For FFHQ, change `--network`; resolution and channel count are read from the
checkpoint:

```bash
python experiments/images/run_edm.py --network /path/edm-ffhq-64x64-uncond-vp.pkl --rule d-grs --branching 2 --lookahead 3 --num-samples 64 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/ffhq-dgrs
```

`--rule target` is the baseline: `speedup` 1.0, one target call per step, and
the reference the speculative rules must match in FID at the same `eps`.

## Many GPUs

`run_edm.py` shards a run across processes: with 4 GPUs each generates a
contiguous quarter of the images, writes its own `shard_XXX.pt`, and rank 0
merges them into one `samples.pt`.

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 experiments/images/run_edm.py --network /path/edm-cifar10-32x32-cond-vp.pkl --rule d-grs --branching 2 --lookahead 3 --num-samples 50000 --num-steps 100 --eps 0.25 --sample-batch 285 --out results/edm/dgrs
```

The sharding workflow provides two guarantees. **Compatible shards are reused**, so a crashed run
restarts at the shard boundary rather than at zero. Every shard carries the run
configuration and global sample range; a mismatch is refused instead of merged
under misleading metadata (`--overwrite` forces regeneration). And **class labels do not depend on the process count**: every
image's label is drawn up front from `--seed` alone and then sliced per rank, so
the same seed at 1 GPU and at 4 conditions on identical labels and the two runs'
FIDs are comparable.

Use `--no-accelerate` for a plain single process, and `--cpu` to force CPU
(needed to run multi-process on a Mac, where accelerate selects MPS and torch
has no MPS `c10d::barrier`).

## Progress

Either form -- one process or many -- reports one bar for the **whole job**:
every rank writes `progress_rankNNN.json` and rank 0 sums them, so a 2-GPU run
shows one line rather than two interleaved ones. Throughput is added over the
ranks still working, which makes the ETA the job's and not one process's.

```
d-grs  [########..............]   37%  18560/50000 img  4.21 img/s  eta 2:04:31  [1:13:29]  2 ranks  spd 1.83  acc 0.712
```

The bar advances *within* a batch as well as between them -- the sampler reports
every round, which is one target call -- so a run of a few large batches still
moves. `spd` and `acc` are the last batch's speedup and acceptance rate.

Redirected to a log, the bar degrades to one line a minute instead of a redrawn
line. `--progress plain` forces that, `--progress bar` forces the bar, and
`--progress none` silences the display; the per-rank JSON files are written
either way, and are deleted by the merge.

## The sweep

```bash
NETWORK=/path/edm-cifar10-32x32-cond-vp.pkl GPUS=0,1,2,3 bash experiments/images/sweep.sh
```

Every `(K, L, rule)` cell whose `samples.pt` and `meta.json` both exist is
skipped, so re-running continues where it stopped.

**`EPS` is a list.** The default, `EPS="0.1 0.3 0.6"`, is the churn grid the
paper reports for CIFAR-10, and the whole `(K, L) x rule` grid — baseline
included — is run once per value, into one output root per value. That is three
times the work of one churn level: pass a single value (`EPS=0.5`) for one
pass. Values are never pooled, because an FID only means anything against the
plain-target arm at the *same* `eps`; the script prints one `fid.py` command
per value, all sharing one real-set cache.

**How the rmc arm is sized.** RMC is a single-proposal coupling and has no tree,
so the sweep hands both rules the same `(K, L)` and `--match` decides the chain
length. The default, `verification`, gives both arms the same **target batch**
`|I|` — the hardware-matched comparison, and `gm_sweep.py`'s default too. At
`K=4, L=3` that is `chain(21)` against a tree with `|I| = 21`, where `budget`
matching would give `chain(84)`. A factor of `K`, so the choice is recorded in
every `meta.json`.

Note this differs from the reference implementation, which budget-matched — its
rmc numbers are not directly comparable with these.

`--sample-batch` is derived per cell as `NODE_BUDGET / |I|`, so memory stays
roughly flat across the grid instead of growing with `K`.

## Configuration

Everything that decides what a run samples can go in one JSON file. The line is
the same one the run signature draws: **protocol goes in the file, placement
stays on the command line**. `--out`, `--device`, `--progress` and the rest are
placement — two runs differing only there produce identical samples — so the
file refuses them, by name, with that reason.

Start from a generated file rather than writing one by hand; it lists every
option that exists now, at its default:

```bash
python experiments/images/run_edm.py --print-config > my-run.json
```

```json
{
  "version": 1,
  "driver": "edm",
  "model":        { "network": null, "edm_repo": null, "toy": false,
                    "toy_resolution": 16, "toy_classes": 0 },
  "schedule":     { "num_steps": 100, "eps": 0.25, "s_noise": 1.0, "shift": 1.0 },
  "method":       { "rule": "d-grs", "branching": 2, "lookahead": 3,
                    "match": "verification" },
  "sampling":     { "seed": 0, "num_samples": 64, "sample_batch": 0 },
  "conditioning": { "labels": "auto" },
  "execution":    { "forward_batch": 0 },
  "sampler":      { "prefetch": "nearest", "evaluate_leaves": false }
}
```

```bash
python experiments/images/run_edm.py --config my-run.json --out results/run
```

**Explicit command-line flags beat the file**, including when the flag repeats
the default — `--eps 0.25` wins over a file saying `0.9`. The run banner prints
which settings the file supplied and which the command line took back, so a
`--config` handed to a script that passes flags explicitly does not look like a
config that was ignored. `version` and `driver` are required: the two drivers
have different defaults (`num_steps` 100 vs 28, `shift` 1.0 vs 3.0), so a file
written for one and read by the other would resolve differently and silently.

An unknown key is an error, not a warning — a key that is read and then ignored
produces a `meta.json` describing a run that did not happen. So is a key in the
wrong section, and the message names the right one.

Whatever the source, the **resolved** settings are what gets recorded: complete,
never the file's subset, in `meta.json` and in the run signature. Two runs that
resolve to the same values have the same signature whether the values came from
a file or from flags, so shards are reusable across the two.

### Schedule options

| option | default | what it does |
| --- | --- | --- |
| `num_steps` | 100 | the horizon `T`. The sampler sees `T` minus the two deterministic endpoints |
| `eps` | 0.25 | churn. Appears squared in the drift correction *and* linearly in the noise, so it moves where a step goes as well as how far it scatters |
| `s_noise` | 1.0 | EDM's `S_noise`. Scales the transition std and **not** the churn mean, so it is not a reparameterisation of `eps`. Must be > 0 |
| `shift` | 1.0 | timestep shift of the sigma grid (SD3.5 ships 3.0) |

### Sampler options

The sampler's own knobs are the `sampler` section above. `--sampler-config` and
`--print-sampler-config` still work on that section alone, and are mutually
exclusive with `--config`:

```bash
SAMPLER_CONFIG=my-sampler.json NETWORK=... bash experiments/images/sweep.sh
# or, for one cell:
python experiments/images/run_edm.py --sampler-config my-sampler.json ...
```

| option | values | what it does |
| --- | --- | --- |
| `prefetch` | `nearest` (default), `parent`, `none` | which already-computed drift the next round's root reuses |
| `evaluate_leaves` | `false` (default), `true` | evaluate the leaf level too, so a fully-accepted round can carry an *exact* drift. `K^L` extra rows per round, and only `prefetch: nearest` can use them |

Those two are the whole list, and the reason is the line the run signature
already draws: a sampler option belongs here when it changes the samples. The
constructor takes three other optional arguments, and none of them qualify.
`check_contract` wraps the verifier in a validating shim without changing what
it returns — it is a debug switch, it stays on the command line as
`--check-contract`, and the signature ignores it. `keep_trajectories` retains
every intermediate state, which the drivers never read and which would cost
`N + 1` states per image to keep. `backend` is an object, resolved from the
array type; there is no JSON for it.

Every key is optional and the file may be omitted entirely; what is *recorded*
is always the complete resolved set, in `meta.json` under `sampler` and in the
run signature. So a shard generated under one policy can never be silently
reused by a run under another — you get the `incompatible (run_signature)`
refusal instead. An unknown key is an error rather than a warning, because a
key that is read and then ignored produces a `meta.json` describing a run that
did not happen.

The sweep checks *finished* cells too, not just shards. A cell is normally
skipped on file existence alone, which says nothing about the settings that
produced it, and neither `sampler` nor `s_noise` appears in a cell's output
path — so both sweeps compare each skipped cell's recorded values against the
ones this run would use, and stop if they differ. A cell with no `sampler` block
predates the option and is read as `prefetch: parent`, `evaluate_leaves: false`
— the only behaviour the batched sampler had then — and a missing `s_noise` is
read as 1.0. An older grid can therefore be continued by asking for those values
explicitly (`SAMPLER_CONFIG=`, `S_NOISE=`).

`prefetch` is worth understanding before changing it. `parent` carries the
verified parent's drift, which was evaluated one step behind the state the next
round actually starts from; `nearest` carries a drift evaluated *at* the
committed step, and costs nothing extra to do so. On the Gaussian mixture at
`d=512, eps=0.06`, `nearest` is worth 35–50% of NFE speedup over `parent`
across the whole `(K, L)` grid, and cuts target rows by about a quarter.
`none` re-evaluates each round's root for an exact drift and pays a target call
per round to do it, which makes it the most expensive of the three and the
slowest.

## What the adapter does

The adapter applies two exact changes of variables.

**Denoiser → velocity.** EDM's network is `D(y; s) = E[x0 | y = x0 + s n]`;
specdiff's trajectory lives on the interpolant `x_t = (1 - t) x0 + t xi`.
Dividing by `(1 - t)` turns one into the other, giving `v = (x_t - D) / t` with
`s = t / (1 - t)`. No retraining, no weight surgery.

**Velocity → transition.** One churn reverse step gives the mean `m^q_n` and a
std that is **independent of the state** — which is exactly the premise the
rank-1 reduction of eqs. (8)–(11) rests on, and why the schedule is a plain
`TabulatedSchedule` rather than a callback into the model.

## Interpreting results

**The horizon is `T`, the speculative window is `T - 2`.** The transition std
vanishes at the first step (`sigma = 1`, where `g` diverges) and the last
(`sigma_next = 0`). Speculation is vacuous at zero churn and `NoiseSchedule`
refuses a non-positive scale (Remark 3), so `build()` exposes only the
stochastic steps and reports the rest, as `experiments/gm/models.py` does.
`sample_trajectory` stitches the two Euler endpoints back on, because here we
want the actual images. Both arms of any comparison pay exactly those two
steps, so leaving them out of the sampler's NFE count is what keeps the
speedup honest.

**Two speedups, and they differ.** `meta.json` reports `speedup`
(`N / target_calls`, what the batch achieved on the clock) and
`mean_isolated_speedup` (what each trajectory would manage alone). One batched
call serves every live trajectory, so the batch advances at the pace of its
slowest member and the first is strictly below the second. The gap is the
straggler cost of sharing a batch — it is what you tune `--sample-batch`
against, and quoting only the second flatters a batched run.

## Checkpoint conditioning and scoring

**Conditional and unconditional are interchangeable.** `--labels auto` (the
default) reads the checkpoint: an `-uncond-` one samples unconditionally, a
`-cond-` one draws a class per image uniformly, which is the class-marginal FID
protocol. So swapping `--network` between them needs no other flag. Use
`--labels 7` to pin one class.

**EDM's conditional networks have no null class.** They were not trained for
classifier-free guidance, so a
`-cond-` checkpoint cannot be run "unconditionally" by omitting the label.
Omitting it makes `EDMPrecond` fall back to a zero embedding, which is not a
trained null token and yields plausible images from the wrong distribution.
`--labels none` on a conditional checkpoint is refused, and so is the reverse.

**Scoring is a separate step**, so one expensive generation run can be measured
many ways. `fid.py` reads the `samples.pt` that `run_edm.py` writes:

```bash
python experiments/images/fid.py --samples results/edm/*/ --dataset cifar10 --num-real 50000 --device cuda:0 --inception-score --output results/edm/fid_report.json
```

Needs `pip install -e '.[fid]'`. It uses torchmetrics'
`FrechetInceptionDistance(feature=2048, normalize=False)` — the same
implementation the reference uses, so the numbers are comparable with the ones
computed there. FID is *not* comparable across Inception implementations, so
that is a fixed choice rather than a detail.

The real-side statistics are cached (`fid_real_<dataset>_<N>_<size>px.pt`).
FID depends on the real images only through `sum(f)`, `sum(f f^T)` and the
count, so the cache stores exact sufficient statistics rather than an approximation.
This makes repeated scoring across a sweep inexpensive. The filename carries the count and resolution, and the cache payload carries a
signature of the actual real-image source. A changed FFHQ directory or archive
is re-featurised instead of silently reusing stale statistics.

## Verification

`tests/test_edm_images.py` runs on CPU without a checkpoint and covers the change of
variables against a velocity derived independently of the denoiser, the
clamping at both singular endpoints, the schedule against
`diffusers.FlowMatchEulerDiscreteScheduler`, state-independence of the std,
`--forward-batch` exactness and its non-effect on NFE, and a KS test that the
speculative sampler reproduces the standard sampler's law.

The adapter can also be checked against the reference implementation
(`accelerating-diffusion-sampling`, `src/models/edm.py` and
`src/schedulers/churn_flow_match_euler.py`) on identical inputs: velocity and
transition std agree exactly, the sigma grid and kernel mean to float32
epsilon. Run `crosscheck_reference.py` with a separate checkout after changing
the transition calculations in either implementation.
