# Tutorials

One notebook per component of the library. They are independent — each defines its own toy model
and imports only from `specdiff` — but the order below is the one that builds up.

| # | notebook | component | what it covers |
| --- | --- | --- | --- |
| 1 | [`tree_tutorial.ipynb`](tree_tutorial.ipynb) | `trees.py` | building a `DraftTree` from a parent list, the uniform/chain families, `B` and `|I|` |
| 2 | [`kernels_tutorial.ipynb`](kernels_tutorial.ipynb) | `kernels.py` | the model interface derived from the paper: reverse SDE → eq. (5) → `TargetTransition` + `NoiseSchedule`, the eq. (24) assumption, the delayed drift (7) and root-drift prefetching, a worked flow-matching conversion (28)–(38), the role of churn (Remark 3), and an index of every equation |
| 3 | [`verifier_tutorial.ipynb`](verifier_tutorial.ipynb) | `verify.py`, `verifiers/rank1.py`, `testing.py` | the one contract, `VerifyRequest`/`VerifyResult`, rank-1 coordinates and degeneracy, `check_exactness` (including what a wrong rule looks like), `CheckedVerifier`, the registry |
| 4 | [`sampler_tutorial.ipynb`](sampler_tutorial.ipynb) | `sampler.py`, `types.py` | Algorithm 3 round by round, `standard_sampler`, `RoundRecord`/`SamplingResult`, truncation near the horizon, cost accounting, projecting speedup from `delta` |
| 5 | [`batched_tutorial.ipynb`](batched_tutorial.ipynb) | `batched.py` | many trajectories per target call, batched proposals, per-row `sigma`, compaction, occupancy and straggler cost |
| 6 | [`backends_tutorial.ipynb`](backends_tutorial.ipynb) | `ops.py` | the array shim, the stack conventions, dtype guards, running the whole sampler on torch |
| 7 | [`end_to_end_tutorial.ipynb`](end_to_end_tutorial.ipynb) | everything | the paper's Gaussian-mixture setting: build a model, measure `delta`, choose a topology under a budget, plan a batch |

## Running them

```bash
conda activate specdiff
pip install -e '.[dev]' jupyter
jupyter lab notebooks
```

`backends_tutorial.ipynb` additionally needs the torch extra (`pip install -e '.[torch]'`).
All outputs in the committed notebooks were produced by running them top to bottom; nothing needs
a GPU and no notebook takes more than a few seconds.

## A note on the verification rules

`specdiff/verifiers/stubs.py` leaves Algorithms 1 (RMC) and 2 (D-GRS) unimplemented on purpose,
so no notebook implements them either. The rules used here are all exact and all *deliberately*
weak:

* `ResampleVerifier` / `DeltaProbe` — never accept; they isolate the sampler and measure the
  headroom a coupling would have (`delta`, and the acceptance probability it implies);
* `AcceptIfIdentical` — accepts only where `delta` is zero to working precision (Remark 2), which
  is enough to exercise the acceptance path and show the sampler's ceiling.

Every speedup number attributed to a real coupling in these notebooks comes from the analytic
cost model, not from a rule. Filling in the stubs is what turns that projection into a
measurement; [`verifier_tutorial.ipynb`](verifier_tutorial.ipynb) §10 has the recipe and the
checklist.
