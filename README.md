# specdiff

Algorithm 3 of *Accelerating Diffusion Sampling via Speculative Draft Trees*: speculative
diffusion sampling over an arbitrary draft tree, with a pluggable verification rule.

The paper's own framing is the design brief — RMC and D-GRS "differ only in the two
components the paper varies: the *draft topology* and the *verification rule*". So those are
the two things you supply, and the sampler knows nothing about either.

```python
from specdiff import DraftTree, DelayedDriftProposal, SpeculativeSampler

sampler = SpeculativeSampler(
    target=my_target,                           # m^q  (the expensive network)
    proposal=DelayedDriftProposal(my_target),   # m^p  (eq. 7)
    schedule=my_schedule,                       # {sigma_n}
    tree=DraftTree.uniform(branching=4, lookahead=3),
    verifier=my_rule,                           # Verify
    num_steps=100,
    check_contract=True,                        # while developing a rule
)
result = sampler.sample(y0, rng=rng)
print(result.summary())   # speedup, NFEs, acceptance rate, batch volume
```

## Install
Create the environment
```bash
conda create -n specdiff python=3.12
conda activate specdiff && cd path-to-specdiff
```
Install
```bash
pip install -e 'specdiff[dev]'
```
Run the test
```bash
python -m pytest tests -q
```
Optional extras. For the PyTorch backend:
```bash
pip install -e 'specdiff[torch]'
```

## Notation

Everything in the code and docs uses these, and nothing else:

| symbol | in code | meaning |
| --- | --- | --- |
| `N` | `num_steps` | denoising steps in the full trajectory — the standard sampler's NFE count |
| `L` | `lookahead`, `tree.depth` | levels of the draft tree below the root; how far ahead one round speculates |
| `K` | `branching`, `tree.branching` | candidate children per node. `K = 1` is RMC's chain |
| `B` | `tree.budget` | states **drafted** per round, `K + ... + K^L` (eq. 12) — the *proposal* budget the paper plots against |
| `\|I\|` | `tree.verification_budget()` | states the **target** evaluates per round, `B / K` when uniform — the *verification* budget, and the batch that must fit in memory |
| `n` | `step`, `start_step` | index of the current step along the trajectory, `0 <= n < N` |
| `sigma_n` | `schedule(step)` | noise scale, shared by proposal and target at step `n` |
| `delta` | `Rank1Frame.delta` | normalised mean mismatch `\|\|mu_q - mu_p\|\| / sigma`; sets every acceptance probability |
| — | `batch`, `batch_size` | independent trajectories (images) sampled at once. Not in the paper; see [Batching over images](#batching-over-images) |

Array shapes are written `(batch, K, *state_shape)`, where `state_shape` is whatever a single
diffusion state is for your model — `(d,)`, `(3, 32, 32)`, `(16, 64, 64)` for an SD3 latent.

## Layout

```
pyproject.toml
specdiff/
  sampler.py  batched.py  trees.py  kernels.py  verify.py  types.py  testing.py  ops.py
  verifiers/
    rank1.py  rmc.py  dgrs.py
tests/       test_sampler.py  test_batched.py  test_rank1.py  test_rmc.py  test_dgrs.py
             test_torch_backend.py
examples/    gaussian_mixture.py
```

| module | responsibility |
| --- | --- |
| `specdiff/sampler.py` | Algorithm 3: drafting, verification, acceptance, NFE accounting |
| `specdiff/batched.py` | the same, over many trajectories per target call |
| `specdiff/trees.py` | draft topologies, layers, internal nodes, truncation `T|_m` |
| `specdiff/kernels.py` | `TargetTransition`, `ProposalTransition`, noise schedules, delayed drift |
| `specdiff/verify.py` | the `Verifier` contract, contract checker, name registry |
| `specdiff/types.py` | `VerifyRequest`/`VerifyResult` and the run records |
| `specdiff/verifiers/rank1.py` | the rank-1 reduction (eqs. 8–11), shared by any isotropic rule |
| `specdiff/verifiers/rmc.py` | Algorithm 1: reflection maximal coupling (`rmc`), `K = 1` |
| `specdiff/verifiers/dgrs.py` | Algorithm 2: greedy rejection sampling (`d-grs`), any `K` |
| `specdiff/testing.py` | statistical exactness test for a rule |
| `specdiff/ops.py` | the only module that touches NumPy/PyTorch |

```bash
pip install -e '.[dev]' && python -m pytest tests -q
```

States must be floating point. An integer array would truncate every Gaussian draw to
zero and hand back a silently wrong trajectory, so `sample()` rejects one outright
rather than letting it through; `float32` and `float64` both work.

## Documentation

| | |
| --- | --- |
| [docs/writing-a-verifier.md](docs/writing-a-verifier.md) | **start here to implement a coupling** — the contract, rank-1 coordinates, exactness testing, the traps |
| [docs/models.md](docs/models.md) | plugging in your own diffusion model, proposals, trees, backends |
| [docs/architecture.md](docs/architecture.md) | how a round works, data flow, cost accounting, design decisions |
| [docs/api-reference.md](docs/api-reference.md) | every exported symbol |
| [notebooks/](notebooks/README.md) | one runnable tutorial per component — trees, kernels, verifiers, the two samplers, backends, and an end-to-end walkthrough |

`examples/gaussian_mixture.py` runs the whole thing end to end on a Gaussian mixture, with no
network involved.

## The one contract

```python
class MyRule(Verifier):
    max_children = None            # or 1 for a single-proposal coupling

    def verify(self, request: VerifyRequest) -> VerifyResult:
        ...
```

`VerifyRequest` gives you `proposal_mean`, `target_mean`, `sigma`, and `children`
(shape `(K, *state_shape)`, **in drafting order**). You return a state, whether it was
accepted, and which child it was. That is the whole interface — a rule written this way
also runs under the batched sampler, unchanged.

Three obligations:

1. `state` is an exact sample from `N(target_mean, sigma^2 I)`.
2. If `accepted`, `state` *is* `children[child_index]` — not a copy-with-correction. The
   sampler descends into that child's subtree, so a mismatch silently corrupts the chain.
3. Children are examined in the order given, if your rule is a sequence coupling.

Obligation 2 is enforced by `check_contract=True`. Obligation 1 cannot be checked per call,
so test it:

```python
from specdiff import check_exactness
report = check_exactness(MyRule(), delta=1.5, num_children=4, seed=0)
assert report.passed, report
```

This projects the returned state onto the displacement direction, where exactness implies
`N(delta, 1)` regardless of what the rule did internally, and runs a KS test. It catches the
coupling bugs that produce plausible-looking but wrong samples.

`seed` controls every draw — the direction, the children, and whatever your rule consumes
via `request.rng` — so a report is reproducible. Keep it: this is a hypothesis test at level
`alpha`, so a *correct* rule fails it about `alpha` of the time, and an irreproducible
failure is indistinguishable from a real coupling bug.

### Working in rank-1 coordinates

`Rank1Frame.from_request(request)` gives you `delta`, the unit `direction`, and
`project`/`reconstruct`. Check `frame.degenerate` before dividing by `delta`:

```python
frame = Rank1Frame.from_request(request)
if frame.degenerate:                    # delta is zero to working precision
    return VerifyResult(request.child(0), accepted=True, child_index=0)
tau = frame.tau(lam)                    # ln(lam) / delta, guarded
```

`degenerate` is `delta <= tol`, not `delta == 0`, with `tol = 1e-10` — a constant, and
deliberately *not* dtype-derived: `delta`, `tau` and the D-GRS masses are Python floats
computed in float64 whatever the states carry. Exact equality is the wrong test: the regime that
breaks a rule is small-and-nonzero `delta`, which is what a *good* proposal produces, and
there `ln(lambda) / delta` saturates `Phi_bar` to exactly 0 and makes the D-GRS residual mass
vanish. Below `tol` the kernels are indistinguishable at the state's own precision, so
accepting unconditionally is the correct limit rather than an approximation.

## Batching over images

Speculation breaks the thing that makes image batching trivial in a standard sampler:
trajectories accept different prefixes and fall out of step immediately. `batched.py`
handles that; the round structure is identical, but each trajectory carries its own `n_i`,
and one target call serves every live trajectory.

```python
from specdiff import BatchedSpeculativeSampler, DelayedDriftProposal

sampler = BatchedSpeculativeSampler(
    target=my_target,
    proposal=DelayedDriftProposal(my_target),   # one frozen drift per trajectory
    schedule=my_schedule,
    tree=DraftTree.uniform(branching=4, lookahead=3),
    verifier=my_rule,                                  # unchanged
    num_steps=100,
    keep_trajectories=False,                           # only the terminal states
)
result = sampler.sample(y0_batch)      # (batch, *state_shape)
print(result.summary())
```

Three things the batch dimension actually changes:

**`sigma` is per row.** `BatchedVerifyRequest` carries `sigmas`, a tuple, because rows of one
verification batch belong to different steps. Use `ops.scale_rows` to broadcast it portably.

**Live rows shrink as the round descends.** A trajectory that rejects at level 1 takes no
part in level 2. The sampler compacts rather than masks, so a rule never sees a dead row and
never needs a validity flag. `request.indices_in_batch[j]` says which image row `j` is, for
rules holding per-image state; `request.row(j)` carries it through as `index_in_batch`.

**Cost is a max, not a mean.** One call serves every live trajectory, so the batch advances
at the pace of its slowest member. `result.speedup` (`N / iterations`) is the wall-clock
number; `result.mean_isolated_speedup` is what those trajectories would each have achieved
alone, and `result.occupancy` is how full the batch stayed. The gap is real and is what you
tune batch size against — the worked example prints ~12% for a batch of 16 at `L=3, alpha=0.84`.
`result.acceptance_rate` is pooled over every trajectory and round, so it is directly
comparable with the single-trajectory result for the same configuration.

Two requirements the batched path adds. The tree must be **level-uniform** (every node at a
given depth has the same width) so that a level's candidates form a rectangular
`(batch, K, *shape)` array — `uniform` and `from_widths` qualify, arbitrary pruned trees do not.
The proposal needs no change at all: `ProposalTransition` takes `indices_in_batch` at every
batch size, so the object you hand the single-trajectory sampler is the object you hand this
one. `DelayedDriftProposal` keeps a `(batch, *shape)` buffer of increments indexed by
`indices_in_batch`, so no trajectory can pick up another's drift, and its warm-ups go in a
single batched call.

Rules get `verify_batch`, whose default implementation loops over rows calling `verify`, so
nothing needs rewriting. Override it when the per-node work is worth vectorising — for the
paper's rules, the sweep over levels `lambda_k` becomes an `(batch,)` vector operation and the
`d`-dimensional projections become single batched ops. An override must stay row-independent:
row `j` may depend only on `request.row(j)`.

`info` carries the same keys either way, with one translation: rows sit at different tree
nodes, so the batched request holds `info["nodes"]` (a tuple) while `request.row(j)` turns it
back into the scalar contract's `info["node"]`. A rule keyed on `info["node"]` therefore runs
unchanged under both samplers.

`check_contract=True` applies identical per-row checks on both paths — shape, finiteness,
index bounds, and accepted-state identity — so a rule the scalar sampler rejects is rejected
under batching too, with the same message plus a row number.

Trajectories share an RNG stream, so a given trajectory is not bit-reproducible across
different batch sizes. Its law is unaffected.

## Design notes

**Why the tree is a first-class object.** Because "the chain is just `K = 1`" is only true if
chain and tree share a representation. They do: a `DraftTree` is a general rooted tree in BFS
order, `DraftTree.chain(L)` is a valid one, and depth-dependent widths (`from_widths`) or
pruned trees need no new code path. Node ids in BFS order also make `T|_m` (eq. 27, needed
when `L_n = min(L, N - n) < L` near the horizon) a slice.

**Why `Verify` takes means and a scale, not distributions.** Eq. (24) is an assumption of the
template, not an implementation detail: proposal and target must be isotropic Gaussians
sharing the variance schedule. Encoding it in the request type means a rule may *rely* on it,
which is what makes the rank-1 reduction legal. `NoiseSchedule.__call__` refuses a zero
scale, since at zero churn both kernels are point masses and speculation is vacuous
(Remark 3).

**Why the proposal has lifecycle hooks.** The interesting proposals keep per-image memory. The
delayed reverse drift needs to know when a round starts and which target drifts have become
available; root-drift prefetching then reuses a drift the previous round already paid for
during verification, instead of spending an extra NFE per round. The hooks let that live in
the proposal rather than as a special case in the sampler. Note the increment is recoverable
from means alone: `gamma * b^q = m^q(y) - y`, so the proposal never needs drift access.

**Where the NFEs are counted.** `TargetTransition.__call__` counts; that is why you call the
instance rather than `.means()`. Cost is reported as `target_calls` (the paper's metric: one
batched call per round) and separately as `target_states_evaluated` (batch volume). The target
is evaluated only at *internal* nodes — leaves are never parents, so `|I| = B / K` for a
uniform tree, which is where a tree buys back some of its verification cost.

## Limitations, and what I would revisit

- **Two samplers, one algorithm.** `sampler.py` and `batched.py` implement the same three
  phases and can drift apart. The scalar one is kept because it is the readable reference and
  the natural thing for a single image; a test asserts the two agree on accounting for a batch of 1.
  If the pair grows a third variant, collapse them and have `SpeculativeSampler` be a
  `batch = 1` façade.
- **Batch occupancy decays.** Trajectories that finish early leave the batch, so late
  iterations run under-full. Refilling with fresh trajectories (continuous batching, as
  serving stacks do for LLMs) would recover it and is a natural next step, since
  `indices_in_batch` is already first-class.
- **Static topology.** The tree is fixed at construction. The paper's closing paragraph wants
  it adapted online to proposal quality and budget; that fits as a `TopologyPolicy` returning
  a tree per round, given the previous round's `RoundRecord`. The sampler already truncates a
  tree per round, so the hook is one line.
- **`check_exactness` is a smoke test.** A KS test on 4k samples catches gross errors, not
  subtle bias in the tail. For a rule you intend to publish, also verify the analytic
  acceptance probability (eq. 15) against the measured one.
- **No `float16` guard.** The rank-1 reduction takes a norm and divides by it; in half
  precision, small `delta` will be noisy. Cast to `float32` for the coupling.
