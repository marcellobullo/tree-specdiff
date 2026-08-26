# FFHQ 64×64 protocol

This protocol runs `edm-ffhq-64x64-uncond-vp.pkl`. It follows
[`cifar10-conditional.md`](cifar10-conditional.md) and documents the FFHQ-specific settings.

## Differences from CIFAR-10

**1. The checkpoint is unconditional.** FFHQ has no
class labels, so EDM publishes only a `-uncond-` checkpoint. `--labels auto`
resolves to `none`; no label configuration is required.

**2. Resolution is read from the checkpoint.** Confirm that the banner reports
`(3, 64, 64)` without an explicit size flag.

**3. You must supply the real images.** `--dataset cifar10` fetches them; FFHQ
does not. `fid.py` needs `--data` pointing at a directory or zip of FFHQ images
at 64×64. Images with other dimensions are resized with bicubic interpolation, so build the
reference set at 64×64 to avoid implicit resizing.
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

## 1. Select a batch size

64×64 is **four times the pixels** of 32×32, so an activation-bound forward
costs roughly four times as much. Expect the batch that fits to be around a
quarter of the CIFAR-10 batch size. Measure memory use with a short run:

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

Use the same `eps` and `--seed` for all three arms.

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

The FFHQ-specific validation is visual: `grid.png` should contain recognizable faces. Inspect
the full-resolution grid because an incorrect change of variables may still produce plausible
texture at thumbnail size.

## Sweeping

```bash
NETWORK=~/edm-ffhq-64x64-uncond-vp.pkl EDM_REPO=~/edm DATASET=ffhq DATA=~/ffhq-64x64.zip GPUS=0,4,5,7 EPS=0.5 NUM_SAMPLES=50000 NODE_BUDGET=500 bash experiments/images/sweep.sh
```

`NODE_BUDGET` is the knob that keeps memory flat across the grid: each cell
gets `--sample-batch = NODE_BUDGET / |I|`. Lower it for 64×64 than you would
for CIFAR-10. `DATASET=ffhq` and `DATA=` only affect the scoring command
printed at the end; generation reads the resolution off the checkpoint.
