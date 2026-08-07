# API reference

Everything exported from `specdiff`. Symbols follow the paper's notation — see the
[notation table](../README.md#notation).

- [Samplers](#samplers) · [Results](#results) · [Draft trees](#draft-trees)
- [Models](#models) · [Proposals](#proposals) · [Batched proposals](#batched-proposals)
- [Verification](#verification) · [Contract types](#contract-types)
- [Rank-1 coordinates](#rank-1-coordinates) · [Testing](#testing) · [Backends](#backends)

---

## Samplers

### `SpeculativeSampler`

```python
SpeculativeSampler(
    target: TargetTransition,
    proposal: ProposalTransition,
    schedule: NoiseSchedule,
    tree: DraftTree,
    verifier: Verifier,
    *,
    num_steps: int,
    check_contract: bool = False,
    backend: Backend | None = None,
)
```

Algorithm 3 over one trajectory. The verifier's topology constraints are checked here, at
construction, not per node.

- `num_steps` — the horizon `N`. Must be `>= 1`.
- `check_contract` — wrap the rule in `CheckedVerifier`. Cheap; leave it on while developing.
- `backend` — only needed for a framework `resolve_backend` does not know.

```python
.sample(init, *, rng=None, on_round=None, record=True) -> SamplingResult
```

Runs from `Y_0 = init`, a single state of shape `state_shape`. The caller draws it from `q_0`,
because the initial distribution is a property of the model, not of the sampler. `init` must be
floating point. `on_round` is a callback receiving each `RoundRecord`; `record=False` skips
retaining them.

### `BatchedSpeculativeSampler`

```python
BatchedSpeculativeSampler(
    target, proposal: BatchedProposal, schedule, tree, verifier,
    *, num_steps: int, check_contract: bool = False,
    keep_trajectories: bool = False, backend: Backend | None = None,
)
```

The same, over `batch_size` independent trajectories at once. Two added requirements: the tree
must be **level-uniform**, and the proposal must be a `BatchedProposal`.

- `keep_trajectories` — retain the full `(batch, N+1, *shape)` history rather than terminal
  states only. Costs memory.

```python
.sample(init, *, rng=None, record=True) -> BatchedSamplingResult
```

`init` has shape `(batch, *state_shape)`. Trajectories are independent but share an RNG stream,
so a given trajectory is **not** bit-reproducible across different batch sizes. Its law is
unaffected.

### `standard_sampler`

```python
standard_sampler(target, schedule, *, num_steps) -> SpeculativeSampler
```

The reference non-speculative Euler–Maruyama loop, as a degenerate case: `K = L = 1` with a
rule that always resamples, so one committed step per target call and `N` NFEs. The denominator
of every speedup number, and a distributional ground truth in tests.

---

## Results

### `SamplingResult`

| attribute | |
| --- | --- |
| `trajectory` | `(N + 1, *state_shape)` — `Y_0` through `Y_N` |
| `sample` | property: the terminal state `Y_N` |
| `rounds` | `tuple[RoundRecord, ...]` |
| `num_steps` | `N` |
| `target_calls` | **NFEs** — one batched call per round, plus any proposal warm-up |
| `target_states_evaluated` | total rows through the target — batch volume, not NFEs |
| `drafted_states` | states the proposal produced |
| `speedup` | property: `num_steps / target_calls` |
| `acceptance_rate` | property: accepted levels / verified levels |
| `summary()` | one-line digest |

### `BatchedSamplingResult`

| attribute | |
| --- | --- |
| `samples` | terminal states, `(batch, *state_shape)` |
| `trajectories` | `(batch, N+1, *shape)` if `keep_trajectories`, else `None` |
| `rounds` | `tuple[BatchedRoundRecord, ...]` |
| `batch_size`, `num_steps` | |
| `target_calls`, `target_states_evaluated`, `drafted_states` | as above |
| `rounds_per_trajectory` | `tuple[int, ...]` |
| `speedup` | property: `N / target_calls` — the **wall-clock** number |
| `mean_isolated_speedup` | property: mean of `N / rounds_i`, what each trajectory would have achieved alone |
| `occupancy` | property: mean fraction of the batch still live per iteration |
| `acceptance_rate` | property: pooled over every trajectory and round |
| `summary()` | |

`speedup` is strictly below `mean_isolated_speedup`: one call serves every live trajectory, so
the batch advances at the pace of its slowest member. The gap is the straggler cost you tune
batch size against.

### `RoundRecord` / `BatchedRoundRecord`

`RoundRecord`: `start_step`, `lookahead` (`L_n`), `committed` (in `[1, L_n]`), `accepted_depth`
(`committed - 1` if rejected, else `committed`), `rejected`, `drafted` (`B_n`), `verified`
(`|I(T_n)|`), `proposals_examined`.

`BatchedRoundRecord`: `iteration`, `active` (trajectory slots), `start_steps`, `committed`,
`accepted_depth`, `rejected`, `drafted`, `verified` — the per-trajectory fields are tuples
aligned with `active`.

---

## Draft trees

### `DraftTree(parents)`

A finite rooted tree `T = (V, pa)`. `parents[0]` must be `-1`; `parents[u] < u` for `u > 0`,
which both rules out cycles and forces the breadth-first ordering the class relies on.

**Constructors**

| | |
| --- | --- |
| `DraftTree.uniform(branching, lookahead)` | the paper's `(K, L)` family, `B = K + ... + K^L` (eq. 12) |
| `DraftTree.chain(lookahead)` | RMC's topology, `K = 1` |
| `DraftTree.from_widths([3, 2])` | depth-dependent widths — 3 children at the root, 2 under each |
| `DraftTree.largest_uniform(budget, branching)` | deepest `(K, L)` tree fitting a proposal budget |

**Inspection**

| | |
| --- | --- |
| `.size` | `\|V\|`, including the root |
| `.budget` | `B = \|V\| - 1`, the drafted (proposal-evaluated) states |
| `.depth` | `L`, the lookahead |
| `.branching` | `K` for a uniform tree; the maximum width otherwise |
| `.internal_nodes` | `I(T)` — nodes with children, and **only** these are target-evaluated |
| `.parent(u)`, `.children(u)`, `.depth_of(u)` | `children` is in drafting order |
| `.layer(level)` | `V_l`, the nodes of depth `level` |
| `.width_at(level)` | children per node at that depth; raises if the level is not uniform |
| `.is_uniform()`, `.is_level_uniform()` | `from_widths([3,1,2])` is level-uniform but not uniform |
| `.truncate(max_depth)` | `T\|_m` (eq. 27) — a slice, cached per instance |

`ROOT` is `0`.

---

## Models

### `TargetTransition`

Abstract. `m^q` — the expensive map. Override `means`; **call the instance**, since `__call__`
keeps the NFE accounting.

```python
means(states, steps) -> Array      # (rows, *shape) at `rows` step indices -> means
__call__(states, steps) -> Array   # counts the call, validates the row count
num_calls, num_states              # counters
reset_stats()                      # called at the start of each sample()
```

Must be a single batched evaluation. `steps` is a tuple of ints and rows may sit at different
steps.

### `NoiseSchedule`

Abstract. Override `sigma(step) -> float`. `__call__` validates and refuses a non-positive
scale (Remark 3). Concrete: `ConstantSchedule(sigma)`, `TabulatedSchedule(sigmas)` — the latter
needs exactly `N` entries.

---

## Proposals

### `ProposalTransition`

Abstract. `m^p` — cheap by assumption, called once per tree level while drafting.

```python
means(states, steps) -> Array                    # required
on_round_start(step, root_state)                 # optional hooks
on_verified(step, state, target_mean)
reset()
stateful: bool = False                           # set True if you cache across calls
```

`stateful` is what stops a single-trajectory proposal from being silently shared across a
batch.

| class | `m^p(y)` |
| --- | --- |
| `IdentityProposal()` | `y` |
| `MirrorProposal(target)` | `m^q(y)` — perfect proposal, `delta = 0` |
| `DelayedDriftProposal(target, *, prefetch=True)` | `y + (m^q(Y~) - Y~)`, eq. (7) |

`prefetch=True` reuses the freshest committed drift, so no extra target call per round;
`prefetch=False` re-evaluates at each round's root, costing one NFE per round for a strictly
better proposal.

---

## Batched proposals

### `BatchedProposal`

Abstract. Like `ProposalTransition`, but `slots` accompanies every call: `slots[i]` is the
trajectory index of row `i`.

```python
means(slots, states, steps) -> Array
on_round_start(slots, steps, roots)
on_verified(slots, steps, states, target_means)
reset(num_slots)
```

| class | |
| --- | --- |
| `StatelessBatchedProposal(inner)` | lifts any stateless `ProposalTransition`. Raises `TypeError` on a stateful one |
| `BatchedDelayedDriftProposal(target, *, prefetch=True)` | native eq. (7); warm-ups collected into **one** batched call |
| `PerSlotProposal(factory, num_slots=None)` | one independent instance per trajectory — correct for anything, at the cost of a Python loop and one warm-up call per slot |

---

## Verification

### `Verifier`

Abstract base for verification rules.

```python
name: str = "verifier"
max_children: int | None = None        # largest K this rule supports; None = unbounded

verify(request: VerifyRequest) -> VerifyResult          # required
verify_batch(request: BatchedVerifyRequest) -> BatchedVerifyResult
__call__(request) -> VerifyResult
supports(num_children) -> bool
check_topology(tree)                   # called once, by the sampler, at construction
reset()                                # drop per-run state
backend_for(request) -> Backend        # static; a backend without importing ops
```

`verify_batch` defaults to a row-wise loop over `verify`, so every rule works under batching
unrewritten. An override must stay **row-independent**: row `j` may depend only on
`request.row(j)`.

See [writing-a-verifier.md](writing-a-verifier.md) for the contract and the obligations.

### `ResampleVerifier`

Always rejects and draws a fresh `Y ~ N(mu_q, sigma^2 I)`. Trivially exact and trivially
useless — it commits one state per target call, reproducing the standard sampler at `1.00x`.
That makes it the sampler's reference point: if Algorithm 3 with this rule does not match a
plain Euler–Maruyama loop in distribution, the bug is in the sampler, not in the coupling.

### `CheckedVerifier(inner)`

Enforces the checkable half of the contract: output type, state shape, finiteness,
`child_index` bounds, and `accepted=True` implying the state *is* the drafted child. Identical
checks on both samplers. Use `check_contract=True` rather than instantiating it directly.

### Registry

```python
@register_verifier("my-rule")          # class decorator; also sets cls.name
class MyRule(Verifier): ...

create_verifier("my-rule", **kwargs) -> Verifier
available_verifiers() -> tuple[str, ...]
```

Registered by the library: `resample`, `rmc`, `d-grs`. All three are registered on
`import specdiff`. `rmc` is Algorithm 1 and is implemented; `d-grs` is Algorithm 2 and is
[left as the reader's work](writing-a-verifier.md#the-papers-two-algorithms).

---

## Contract types

### `VerifyRequest`

Everything a rule is allowed to see at one node.

| field | |
| --- | --- |
| `step` | `n + \|u\|` — the transition being verified |
| `proposal_mean`, `target_mean`, `sigma` | `m^p(Y_u)`, `m^q(Y_u)`, `sigma_step` |
| `children` | `(K, *state_shape)`, **in drafting order** |
| `parent_state` | `Y_u`; optional |
| `slot` | trajectory index — always `0` under the scalar sampler |
| `rng` | the run's generator. Rules must use this one |
| `info` | free-form; carries `"level"` and `"node"` |
| `.num_children`, `.state_shape`, `.child(i)` | properties/helpers |

### `VerifyResult`

`state`, `accepted`, `child_index=None`, `proposals_examined=None`. `__post_init__` rejects
`accepted` without a `child_index`, and a `child_index` on a rejection.

### `BatchedVerifyRequest`

`VerifyRequest` with a leading batch dimension and one substantive difference: **`sigmas` is
per row**, because rows belong to different steps. Also carries `slots`; `children` is
`(batch, K, *shape)`. All rows are live — the sampler compacts rather than masks.

`.row(j)` extracts row `j` as a `VerifyRequest`, translating the batch-wide `info["nodes"]`
tuple back into the scalar contract's `info["node"]`.

### `BatchedVerifyResult`

`states` `(batch, *shape)`, `accepted`, `child_index`, `proposals_examined` — tuples, validated
row-wise. `BatchedVerifyResult.from_rows(results, ops)` builds one from a list of
`VerifyResult`.

---

## Rank-1 coordinates

`from specdiff.verifiers.rank1 import Rank1Frame, DEGENERATE_TOL`

```python
Rank1Frame.from_request(request, *, tol=None) -> Rank1Frame
```

| | |
| --- | --- |
| `.delta` | `\|\|mu_q - mu_p\|\| / sigma` — controls every acceptance probability in the paper |
| `.direction` | the unit vector `e` |
| `.tol` | degeneracy tolerance; defaults to `sqrt(eps)` of the state dtype |
| `.degenerate` | `delta <= tol`, **not** `delta == 0` — see [why](writing-a-verifier.md#degeneracy) |
| `.project(state)` | `Y -> (S, Z_perp)`, eq. (10) |
| `.reconstruct(s, z_perp)` | `(S, Z_perp) -> Y`, eq. (11) |
| `.tau(level)` | `ln(level) / delta`, Appendix B.2. Raises `ZeroDivisionError` on a degenerate frame |

Set the module-level `DEGENERATE_TOL` to override the default globally.

---

## Testing

```python
check_exactness(
    verifier, *, delta=1.0, num_children=2, dim=8,
    num_samples=4000, alpha=0.01, seed=0, array_like=None,
) -> ExactnessReport
```

Runs a rule on a synthetic node `num_samples` times and KS-tests the projection of its output,
which under exactness is `N(delta, 1)` regardless of what the rule did internally.

`delta=0.0` is supported and worth testing: the harness projects onto the direction it
constructed the node from, so the statistic stays meaningful where the mean displacement
vanishes. For a sweep over many `(delta, K)` cells, tighten `alpha` — more tests means more
chances for a correct rule to trip one.

`seed` controls **every** draw, including whatever the rule consumes via `request.rng`, so a
report is reproducible. Keep it: this is a level-`alpha` test, so a correct rule fails about
`alpha` of the time, and an irreproducible failure cannot be told apart from a real bug.

`ExactnessReport`: `delta`, `num_children`, `num_samples`, `ks_statistic`, `critical_value`,
`acceptance_rate`, `mean_examined`, `.passed`, `str()`.

---

## Backends

`ops.py` is the only module that touches NumPy or PyTorch.

```python
resolve_backend(reference) -> Backend      # picks NumPy or PyTorch from an example array
```

`Backend` is abstract with 17 methods: `zeros_stack`, `randn_stack`, `uniform`, `make_rng`,
`take`, `put`, `repeat_rows`, `stack_rows`, `scale_rows`, `group_rows`, `norm`, `dot`, `copy`,
`is_finite`, `is_floating`, `finfo_eps`, `allclose`. Concrete helpers: `numel`,
`check_state_dtype`.

State arrays are always stacks of shape `(num_nodes, *state_shape)`. Node and step indices are
plain Python ints; nothing framework-specific crosses the public API except the state arrays.

Module-level helpers: `default_state(dim)`, `standard_normal_cdf(x)`, `standard_normal_sf(x)` —
the last two so verifiers and tests need no SciPy.

See [models.md](models.md#adding-a-backend) for writing one.
