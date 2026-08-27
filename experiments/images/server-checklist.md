# Environment validation

Run these checks after configuring a new GPU environment or downloading a checkpoint. Each
section provides a command and the expected result before starting a full experiment.

## Automated CPU validation

The test suite covers the following behavior with the toy denoiser:

- the denoiser→velocity change of variables, against a velocity derived
  independently of the denoiser (exact to 1e-12 in float64);
- the churn schedule against `diffusers.FlowMatchEulerDiscreteScheduler` **and**
  against the reference implementation's `ChurnFlowMatchEulerScheduler`
  (velocity and transition std agree exactly; sigma grid and kernel mean to
  float32 epsilon);
- the two deterministic Euler endpoints, and that both arms pay exactly two;
- `--forward-batch` exactness and its non-effect on the NFE count;
- that the speculative sampler reproduces the standard sampler's law (KS test);
- per-image class labels routed correctly through the batched sampler;
- shard arithmetic, shard reuse, merge, and a real **two-process** run (on CPU).

Run it with:

```bash
python -m pytest tests -q
```

## Environment-dependent validation

### 1. Loading a real checkpoint

Unpickling requires EDM's `torch_utils` and `dnnlib` modules and executes code stored in the
pickle. Use checkpoints from a trusted source.

```bash
specdiff-download-edm
curl -L -o edm/edm-cifar10-32x32-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

```bash
python experiments/images/run_edm.py --network edm/edm-cifar10-32x32-uncond-vp.pkl --no-accelerate --rule target --num-samples 8 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-target
```

Expect, in order:

- the banner says `(3, 32, 32) states, T=100 (98 speculative + 2 Euler)` —
  resolution is read off the checkpoint, not from a flag;
- `labels: none` (this is the uncond checkpoint);
- `"target_calls": 98`, `"speedup": 1.0`, `"end_to_end_speedup": 1.0`. The
  baseline must be exactly one call per speculative step. Anything else means
  the endpoint bookkeeping is wrong.

### 2. Speculation on real weights

```bash
python experiments/images/run_edm.py --network edm/edm-cifar10-32x32-uncond-vp.pkl --no-accelerate --rule d-grs --branching 2 --lookahead 3 --num-samples 64 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-dgrs
```

Expect `speedup > 1`, an `acceptance_rate` strictly between 0 and 1, and
`grid.png` that looks like CIFAR-10 rather than noise. A speedup of exactly 1.0
means nothing is being accepted; an acceptance rate of 1.0 means the verifier is
accepting everything, which would be a coupling bug.

Then the rmc arm, which should report `"match": "verification"` and
`"chain_depth": 7` for these `(K, L)`:

```bash
python experiments/images/run_edm.py --network edm/edm-cifar10-32x32-uncond-vp.pkl --no-accelerate --rule rmc --branching 2 --lookahead 3 --num-samples 64 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-rmc
```

### 3. FFHQ checkpoint metadata

```bash
curl -L -o edm/edm-ffhq-64x64-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl
python experiments/images/run_edm.py --network edm/edm-ffhq-64x64-uncond-vp.pkl --no-accelerate --rule d-grs --branching 2 --lookahead 3 --num-samples 16 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-ffhq
```

Expect `(3, 64, 64)` in the banner with no other flag changed, and faces in
`grid.png`.

### 4. Conditional checkpoint

Use this check to validate one-hot conditioning with a pretrained `EDMPrecond`.

```bash
curl -L -o edm/edm-cifar10-32x32-cond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
python experiments/images/run_edm.py --network edm/edm-cifar10-32x32-cond-vp.pkl --no-accelerate --rule d-grs --branching 2 --lookahead 3 --num-samples 64 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-cond
```

Expect `labels: uniform over 10 classes` and `"conditional": true`. Sanity
check the guard too — this must be **refused**, not silently sampled:

```bash
python experiments/images/run_edm.py --network edm/edm-cifar10-32x32-cond-vp.pkl --no-accelerate --labels none --rule d-grs --num-samples 4 --num-steps 100 --out results/edm/should-fail
```

For an additional conditioning check, generate with `--labels 3` and confirm
that the grid contains a single visible class.

### 5. Multi-GPU

Use this check to validate NCCL, `--multi_gpu`, and per-rank CUDA placement.

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 experiments/images/run_edm.py --network edm/edm-cifar10-32x32-cond-vp.pkl --rule d-grs --branching 2 --lookahead 3 --num-samples 256 --num-steps 100 --eps 0.25 --sample-batch 64 --out results/edm/smoke-4gpu
```

Expect four `rank N: images a..b` lines covering `0..255` with no gaps or
overlaps, `"num_processes": 4`, `"num_samples": 256`, and no `shard_*.pt` left
behind after the merge. Then re-run the same command: it should print
`reusing shard_00N.pt` for every rank and finish almost immediately.

There is a `barrier()` helper in `run_edm.py` with a fallback for a Mac-only
torch bug (no MPS `c10d::barrier`). On CUDA the first call should always
succeed. If the fallback is used on CUDA, inspect the process-group configuration.

### 6. The sweep

Start with a small sweep to validate GPU discovery and the nested
`accelerate launch --multi_gpu` command:

```bash
NETWORK=edm/edm-cifar10-32x32-cond-vp.pkl GPUS=0,1,2,3 EPS=0.5 NUM_SAMPLES=256 CONFIGS="2,2 2,3" bash experiments/images/sweep.sh
```

Expect a per-cell `|I|` table, the plain-target baseline first, then each cell.
Kill it mid-cell and re-run: finished cells must be skipped and the interrupted
cell must resume from its shards.

### 7. Reference cross-check

```bash
python experiments/images/crosscheck_reference.py --reference-repo <accelerating-diffusion-sampling checkout>
```

Expect `OK: agreement within 1e-06`.
