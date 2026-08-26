# Server checklist — what is still unverified

Everything in `experiments/images/` was built and tested on a laptop with no
GPU, against `toy.py`'s closed-form stand-in denoiser. This file lists what that
could **not** cover, in rough order of risk, with the command for each and what
a correct result looks like.

## Already verified — do not redo

On CPU, with the toy denoiser, all of this passes (`pytest tests -q`, 120 tests):

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

## Still unverified

### 1. Loading a real checkpoint

The single largest unknown. Unpickling needs EDM's `torch_utils` and `dnnlib`
importable, and **executes code from the pickle**.

```bash
git clone https://github.com/NVlabs/edm.git ~/specdiff/edm
curl -L -o ~/specdiff/edm/edm-cifar10-32x32-cond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

```bash
python experiments/images/run_edm.py --network ~/edm-cifar10-32x32-uncond-vp.pkl --edm-repo ~/edm --no-accelerate --rule target --num-samples 8 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-target
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
python experiments/images/run_edm.py --network ~/edm-cifar10-32x32-uncond-vp.pkl --edm-repo ~/edm --no-accelerate --rule d-grs --branching 2 --lookahead 3 --num-samples 64 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-dgrs
```

Expect `speedup > 1`, an `acceptance_rate` strictly between 0 and 1, and
`grid.png` that looks like CIFAR-10 rather than noise. A speedup of exactly 1.0
means nothing is being accepted; an acceptance rate of 1.0 means the verifier is
accepting everything, which would be a coupling bug.

Then the rmc arm, which should report `"match": "verification"` and
`"chain_depth": 7` for these `(K, L)`:

```bash
python experiments/images/run_edm.py --network ~/edm-cifar10-32x32-uncond-vp.pkl --edm-repo ~/edm --no-accelerate --rule rmc --branching 2 --lookahead 3 --num-samples 64 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-rmc
```

### 3. FFHQ — the "same command, one path changed" claim

```bash
curl -L -o ~/specdiff/edm/edm-ffhq-64x64-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl
python experiments/images/run_edm.py --network ~/specdiff/edm/edm-ffhq-64x64-uncond-vp.pkl --edm-repo ~/specdiff/edm --no-accelerate --rule d-grs --branching 2 --lookahead 3 --num-samples 16 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-ffhq
```

Expect `(3, 64, 64)` in the banner with no other flag changed, and faces in
`grid.png`.

### 4. The conditional path against real weights

Only ever exercised against the toy network — the one-hot plumbing has never
met a real `EDMPrecond`.

```bash
curl -L -o ~/specdiff/edm/edm-cifar10-32x32-cond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
python experiments/images/run_edm.py --network ~/specdiff/edm/edm-cifar10-32x32-cond-vp.pkl --edm-repo ~/specdiff/edm --no-accelerate --rule d-grs --branching 2 --lookahead 3 --num-samples 64 --num-steps 100 --eps 0.25 --device cuda:0 --out results/edm/smoke-cond
```

Expect `labels: uniform over 10 classes` and `"conditional": true`. Sanity
check the guard too — this must be **refused**, not silently sampled:

```bash
python experiments/images/run_edm.py --network ~/specdiff/edm/edm-cifar10-32x32-cond-vp.pkl --edm-repo ~/edm --no-accelerate --labels none --rule d-grs --num-samples 4 --num-steps 100 --out results/edm/should-fail
```

A stronger check if you want one: generate with `--labels 3` and confirm the
grid is visibly one class.

### 5. Multi-GPU

Only ever run as two CPU processes. NCCL, `--multi_gpu`, and per-rank device
placement are untested.

```bash
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 experiments/images/run_edm.py --network ~/specdiff/edm/edm-cifar10-32x32-cond-vp.pkl --edm-repo ~/specdiff/edm --rule d-grs --branching 2 --lookahead 3 --num-samples 256 --num-steps 100 --eps 0.25 --sample-batch 64 --out results/edm/smoke-4gpu
```

Expect four `rank N: images a..b` lines covering `0..255` with no gaps or
overlaps, `"num_processes": 4`, `"num_samples": 256`, and no `shard_*.pt` left
behind after the merge. Then re-run the same command: it should print
`reusing shard_00N.pt` for every rank and finish almost immediately.

There is a `barrier()` helper in `run_edm.py` with a fallback for a Mac-only
torch bug (no MPS `c10d::barrier`). On CUDA the first call should always
succeed, so **the fallback should never fire** — if you see it engage, something
is wrong with the process group.

### 6. The sweep

`nvidia-smi` preflight and `accelerate launch --multi_gpu` inside the script are
untested. Start small:

```bash
NETWORK=~/specdiff/edm/edm-cifar10-32x32-cond-vp.pkl EDM_REPO=~/edm GPUS=0,1,2,3 NUM_SAMPLES=256 CONFIGS="2,2 2,3" bash experiments/images/sweep.sh
```

Expect a per-cell `|I|` table, the plain-target baseline first, then each cell.
Kill it mid-cell and re-run: finished cells must be skipped and the interrupted
cell must resume from its shards.

### 7. Port fidelity, on the real environment

```bash
python experiments/images/crosscheck_reference.py --reference-repo <accelerating-diffusion-sampling checkout>
```

Expect `OK: agreement within 1e-06`.

### 8. FID — not ported

specdiff has no scorer. `samples.pt` is written in exactly the layout the
reference implementation's `scripts/fid_from_samples.py` reads (uint8
`(N, C, H, W)`), so use that one; it also keeps the numbers comparable with
FIDs already computed there. Setting `FID_REPO=<checkout>` makes `sweep.sh`
print the command.

**Note on comparability:** the rmc arm here is *verification*-matched
(`chain(|I|)`), while the reference implementation budget-matched (`chain(B)`).
The d-grs numbers are comparable; the rmc ones are not.
