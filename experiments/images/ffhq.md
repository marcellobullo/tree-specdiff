# FFHQ 64×64 — the full protocol

End-to-end run on `edm-ffhq-64x64-uncond-vp.pkl`. Structurally identical to
[`cifar10-conditional.md`](cifar10-conditional.md); this file records only what
differs, and there are three things.

## The three differences

**1. It is unconditional, and there is no choice about that.** FFHQ has no
class labels, so EDM publishes only a `-uncond-` checkpoint. `--labels auto`
resolves to `none` on its own — nothing to pass, nothing to match against a
conditional baseline.

**2. 64×64, read from the checkpoint.** The command is otherwise the *same* as
CIFAR-10's with one path changed; resolution and channel count come off the
`.pkl`. The banner printing `(3, 64, 64)` with no size flag is the check that
this works.

**3. You must supply the real images.** `--dataset cifar10` fetches them; FFHQ
does not. `fid.py` needs `--data` pointing at a directory or zip of FFHQ images
at 64×64, and images of another size are resized bicubic rather than refused —
so build the reference set at the right resolution rather than relying on that.
The reference implementation used an `ffhq-64x64.zip`; if you already have one
from that repo, point at it and your numbers stay comparable.

Everything else — the horizon, the matching, the arms, the checks — is as in
the CIFAR-10 file. Read that one first.

## 0. Prerequisites

```bash
pip install -e '.[all]'
git clone https://github.com/NVlabs/edm.git ~/edm
curl -L -o ~/edm-ffhq-64x64-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl
```

Plus the real set, as a directory or zip of 64×64 images — call it
`~/ffhq-64x64.zip` below.

## 1. Size the batch — lower than CIFAR-10

64×64 is **four times the pixels** of 32×32, so an activation-bound forward
costs roughly four times as much. Expect the batch that fits to be around a
quarter of what CIFAR-10 tolerated. Probe rather than assume:

```bash
python experiments/images/run_edm.py --network ~/edm-ffhq-64x64-uncond-vp.pkl --edm-repo ~/edm --no-accelerate --rule d-grs --branching 2 --lookahead 3 --num-samples 128 --num-steps 100 --eps 0.5 --sample-batch 16 --device cuda:4 --out /tmp/probe-ffhq
```

Read the `memory:` line and the `img/s`, then adjust. `--forward-batch` caps
the activation peak without changing the batch or the results, and matters more
here than at 32×32.

## 2. Generate the three arms

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,4,5,7 experiments/images/run_edm.py --network ~/edm-ffhq-64x64-uncond-vp.pkl --edm-repo ~/edm --rule target --num-samples 50000 --num-steps 100 --eps 0.5 --seed 0 --sample-batch 64 --out results/ffhq/plain-target
```

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,4,5,7 experiments/images/run_edm.py --network ~/edm-ffhq-64x64-uncond-vp.pkl --edm-repo ~/edm --rule d-grs --branching 2 --lookahead 3 --num-samples 50000 --num-steps 100 --eps 0.5 --seed 0 --sample-batch 16 --out results/ffhq/K2_L3/d-grs
```

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,4,5,7 experiments/images/run_edm.py --network ~/edm-ffhq-64x64-uncond-vp.pkl --edm-repo ~/edm --rule rmc --branching 2 --lookahead 3 --num-samples 50000 --num-steps 100 --eps 0.5 --seed 0 --sample-batch 16 --out results/ffhq/K2_L3/rmc
```

Same `eps` and `--seed` across all three, as always.

## 3. Score

```bash
python experiments/images/fid.py --samples results/ffhq/plain-target results/ffhq/K2_L3/d-grs results/ffhq/K2_L3/rmc --dataset ffhq --data ~/ffhq-64x64.zip --num-real 50000 --device cuda:4 --inception-score --cache-dir results/ffhq --output results/ffhq/fid_report.json
```

FFHQ has 70k images, so `--num-real 50000` takes the first 50,000 in sorted
filename order. That is deterministic and reproducible, but it is *a* choice —
if you have existing FFHQ numbers, check they used the same count, since FID
against a different real N is a different number. The cache filename records
the count, so the two cannot be confused.

## 4. What the results must show

Identical to CIFAR-10: the three FIDs must agree within sampling noise (the
speculative rules sample the same law as plain target at temperature 1),
speedup above 1, acceptance strictly between 0 and 1.

The one FFHQ-specific check is visual — `grid.png` should be faces. At 64×64 a
broken change of variables produces plausible-looking texture that survives a
glance at a thumbnail, so look at it properly.

## Sweeping

```bash
NETWORK=~/edm-ffhq-64x64-uncond-vp.pkl EDM_REPO=~/edm DATASET=ffhq DATA=~/ffhq-64x64.zip GPUS=0,4,5,7 EPS=0.5 NUM_SAMPLES=50000 NODE_BUDGET=500 bash experiments/images/sweep.sh
```

`NODE_BUDGET` is the knob that keeps memory flat across the grid: each cell
gets `--sample-batch = NODE_BUDGET / |I|`. Lower it for 64×64 than you would
for CIFAR-10. `DATASET=ffhq` and `DATA=` only affect the scoring command
printed at the end; generation reads the resolution off the checkpoint.
