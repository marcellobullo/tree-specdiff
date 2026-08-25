# CIFAR-10, class-conditional — the full protocol

End-to-end run on `edm-cifar10-32x32-cond-vp.pkl`: generate, then score FID and
Inception Score. This is the experiment; [`server-checklist.md`](server-checklist.md)
is the smoke test that should pass before you start it.

## What is fixed, and what is yours to choose

Fixed by the protocol:

| | |
| --- | --- |
| checkpoint | `edm-cifar10-32x32-cond-vp.pkl` (conditional, `label_dim = 10`) |
| horizon | `--num-steps 100` (98 speculative + 2 deterministic Euler endpoints) |
| samples | 50,000 |
| real set | 50,000 CIFAR-10 **train** images |
| labels | `--labels auto` → one uniform class per image, i.e. the class marginal |
| matching | `--match verification`: the rmc chain gets the same target batch `\|I\|` as the tree |

**`eps` is a parameter, not a constant.** The reference implementation swept
`0.1`, `0.3`, `0.5`, `0.6` in different scripts; its FID script defaulted to
`0.5`. Pick the value you intend to report and keep it fixed across every arm —
comparing arms at different `eps` compares nothing.

## 0. Prerequisites

```bash
pip install -e '.[all]'
git clone https://github.com/NVlabs/edm.git ~/edm
curl -L -o ~/edm-cifar10-32x32-cond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

Find free GPUs — this matters, wall clock is set by the slowest rank:

```bash
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv
```

## 1. Size the batch before committing hours

`--sample-batch` sets memory *and* throughput: one target call carries
`sample_batch × |I|` states, and `|I| = 7` for `K=2, L=3`. Too small and the
GPU idles between launches. Probe it:

```bash
python experiments/images/run_edm.py --network ~/edm-cifar10-32x32-cond-vp.pkl --edm-repo ~/edm --no-accelerate --rule d-grs --branching 2 --lookahead 3 --num-samples 256 --num-steps 100 --eps 0.5 --sample-batch 64 --device cuda:4 --out /tmp/probe
```

Read the `memory:` line in the banner and the `img/s` in the progress line.
Double `--sample-batch` until it stops helping or stops fitting. If a large
batch is near the edge, keep it and add `--forward-batch` — it caps the
activation peak and is exact (same RNG, same accept/reject, same NFE), just
keep it fixed across every arm.

## 2. Generate the three arms

`GPUS` should be the free ones. Use the same `--seed` everywhere: labels are
drawn from the seed alone and are independent of the process count, so every
arm then conditions on **identical** classes and the FIDs are paired.

**Baseline** — this is the reference FID the speculative arms must match:

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,4,5,7 experiments/images/run_edm.py --network ~/edm-cifar10-32x32-cond-vp.pkl --edm-repo ~/edm --rule target --num-samples 50000 --num-steps 100 --eps 0.5 --seed 0 --sample-batch 256 --out results/cifar10-cond/plain-target
```

**D-GRS** (the tree):

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,4,5,7 experiments/images/run_edm.py --network ~/edm-cifar10-32x32-cond-vp.pkl --edm-repo ~/edm --rule d-grs --branching 2 --lookahead 3 --num-samples 50000 --num-steps 100 --eps 0.5 --seed 0 --sample-batch 64 --out results/cifar10-cond/K2_L3/d-grs
```

**RMC** (the chain, verification-matched to that tree):

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,4,5,7 experiments/images/run_edm.py --network ~/edm-cifar10-32x32-cond-vp.pkl --edm-repo ~/edm --rule rmc --branching 2 --lookahead 3 --num-samples 50000 --num-steps 100 --eps 0.5 --seed 0 --sample-batch 64 --out results/cifar10-cond/K2_L3/rmc
```

Note rmc takes the *same* `--branching 2 --lookahead 3`: it has no tree of its
own, and `--match verification` turns that pair into `chain(7)`. Its
`meta.json` records `chain_depth`.

Runs are resumable — re-issuing the same command reuses finished shards.

## 3. Score

```bash
python experiments/images/fid.py --samples results/cifar10-cond/plain-target results/cifar10-cond/K2_L3/d-grs results/cifar10-cond/K2_L3/rmc --dataset cifar10 --num-real 50000 --device cuda:4 --inception-score --cache-dir results/cifar10-cond --output results/cifar10-cond/fid_report.json
```

The first invocation featurises 50k real images; `--cache-dir` means every
later arm reuses those statistics exactly, so scoring more cells is nearly free.

## 4. What the results must show

**The three FIDs must agree within sampling noise.** This is the scientific
check, not a nicety: at temperature 1 the speculative rules sample *the same
law* as plain target sampling, so a real FID gap means a coupling bug, not a
worse method. At 50k samples FID noise is roughly ±0.1; a gap of several points
is a bug.

**Speedup must exceed 1 for both speculative arms**, and `acceptance_rate`
must sit strictly between 0 and 1. An acceptance of 1.0 means the verifier is
accepting everything — also a bug.

Quote **`speedup`** (over the 98-step speculative window) or
**`end_to_end_speedup`** (including the two Euler endpoints), but say which.
`mean_isolated_speedup` is what a trajectory would achieve alone; the gap to
`speedup` is the straggler cost of sharing a batch.

Check `seconds_per_rank` in each `meta.json`. A large spread means one GPU was
contended and the run waited on it — the throughput number, not the NFE
numbers, is what that invalidates.

## Sweeping (K, L) instead

For the grid rather than one cell:

```bash
NETWORK=~/edm-cifar10-32x32-cond-vp.pkl EDM_REPO=~/edm GPUS=0,4,5,7 EPS=0.5 NUM_SAMPLES=50000 bash experiments/images/sweep.sh
```

It runs the baseline first, then every `(K, L)` cell for both rules, sizes
`--sample-batch` per cell from `NODE_BUDGET / |I|` so memory stays flat as `K`
grows, skips finished cells, and prints the `fid.py` command at the end. Start
with `NUM_SAMPLES=256 CONFIGS="2,2 2,3"` to confirm it behaves before letting
it loose.

## Comparability

The rmc arm here is **verification**-matched. The reference implementation
budget-matched (`chain(B)`, a factor of `K` longer). D-GRS numbers are
comparable with theirs; rmc numbers are not. Use `MATCH=budget` if you need
theirs.
