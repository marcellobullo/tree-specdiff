# Experiments

Paper replications. These are long-running and produce artifacts, which is why
they live here rather than in `examples/` (seconds, pedagogical) or `tests/`
(correctness).

| directory | setting |
| --- | --- |
| `gm/` | the Gaussian mixture of Section 5.1 — no network, runs on a laptop |
| `images/` | pretrained EDM checkpoints (CIFAR-10, FFHQ) — see [images/README.md](images/README.md) |

## Gaussian mixture — the (K, L) sweep (Figures 1 and 3)

```bash
RUN=results/gm/$(date +%Y%m%d-%H%M%S)
python experiments/gm/gm_sweep.py --out $RUN --lazy --n-workers 8   # raw.csv, resumable
python experiments/gm/plot_gm.py  --out $RUN                        # summary.csv + figures
```

`plot_gm.py` produces both paper figures from the same reduction:

| file | |
| --- | --- |
| `figure1_frontier.{pdf,png}` | speed-up against the verification budget `B_ver` — the efficiency frontier |
| `figure3_grid_speedup.{pdf,png}` | the `(K, L)` grid, speed-up per cell, one panel per rule |
| `figure3_grid_calls.{pdf,png}` | the same grid in raw target calls |

One plotting script rather than two, because both figures come from the same
`raw.csv -> summary.csv` reduction; splitting them would duplicate that step or
make one import the other.

Defaults reproduce the reference run: `d=512`, 5 components, `T=30`,
`eps=0.06`, mixture seed `20260714`, `K, L in 1..7`, 100 trajectories per cell.

`gm_sweep.py` writes each cell as it completes and skips cells already present,
so an interrupted run resumes.

### `--lazy` and `--n-workers`

`--lazy` swaps the eager sampler for the cost simulator in `lazy.py`, which
realises only the committed branch. Eagerly, `(7,7)` needs 7.9 GB and is
unrunnable; lazily it is `K * L = 49` states per round. **The full 7x7 grid,
100 replicates, `--lazy --n-workers 8`: 3.3 seconds**, with no cells skipped.

The simulator is not a sampler — it counts one call per round while evaluating
means on demand, which is right for NFEs and wrong for wall-clock, and its
`target_rows` is analytic rather than measured. `validate_lazy.py` is the gate:
36 configurations of rule x (K,L) x `prefetch` x `evaluate_leaves`, worst
|z| = 1.51 on mean `target_calls`.

`--n-workers` parallelises replicates within each cell. Results are
**byte-identical at any worker count** — streams are keyed by
`(K, L, replicate)` rather than drawn in sequence, so nothing depends on
evaluation order. Verified by hashing the sorted CSV at 1 and 6 workers, in both
modes. BLAS thread counts are pinned to 1 in the workers; without that, NumPy
oversubscribes the cores and parallel is *slower* than serial.

One caveat under `--no-lazy`: each worker holds its own copy of the current
cell's tree and states, so memory scales with worker count. Lower
`--max-verification-budget` accordingly, or use `--lazy`. It also writes `config.json` next to the results
and refuses to append to a directory produced by a different configuration —
the reference implementation computed its config and discarded it, which is why
recovering what produced a given results directory meant digging through shell
history. Cells whose verification batch exceeds
`--max-internal` are skipped and reported; at `d=512` the top corner is
memory-bound (`(7,7)` needs ~20 GB, `(6,7)` ~7 GB), everything else fits.

### Two budgets

`DraftTree` separates them, because they are paid in different currencies:

| | | |
| --- | --- | --- |
| `tree.budget` | `B = K + ... + K^L` | states **drafted** — eq. (12), the figure's x-axis |
| `tree.verification_budget()` | `\|I\| = B / K` | states the **target** evaluates — the batch that must fit in memory |
| `tree.verification_budget(evaluate_leaves=True)` | `B + 1` | as above plus the leaf level, `K^L` extra rows |

Under the self-speculative delayed drift, drafting a state is a vector add — so
`B` is not the hardware requirement. Reading it as one overstates what the
target sees by a factor of `K`, which is precisely why a wide tree is
affordable. (The `+1` is the root, which specdiff evaluates because its target
mean verifies the depth-1 children; an implementation whose carry is exact *at*
the root can skip it and land on exactly `B`.)

### Matching the two arms

Each D-GRS cell runs `DraftTree.uniform(K, L)`; its RMC counterpart runs a chain
of the same proposal budget `B`. Both `target_calls` (NFEs) and `target_rows`
are recorded per trajectory:

| metric | what it assumes |
| --- | --- |
| `target_calls` | one batched call per round is one unit of cost — latency-bound, batch effectively free |
| `target_rows` | every state through the target costs — throughput- or memory-bound |

The RMC arm's `target_rows` saturates near the horizon while D-GRS's keeps
growing with `K`. That is not an artefact of the protocol to be corrected: a
round starting at step `n` truncates to `min(depth, N - n)`, so a **chain
cannot spend a budget deeper than the trajectory is long**. Give RMC more
hardware and it has nowhere to put it. Absorbing that budget through width
instead is exactly what the tree topology adds, and the gap between the two
`target_rows` columns is a measure of it rather than a caveat against it.

Choosing `\|I\|` instead of `B` as the matched quantity was measured and changes
almost nothing — at most 0.13x at `(K=2, L=2)`, and nothing at all in most
cells, since both clamp to the horizon once `\|I\| >= N`.

### Leaves

Only internal nodes are verified, so `verification_budget()` is `|I| = B / K`
and the leaf level is never evaluated. Evaluating it would let the delayed-drift
proposal carry an *exact* drift on full acceptance, worth a few percent of NFE
speedup — at `K^L` extra rows per round. Measured on the reference
implementation, the trick buys a roughly constant ~7% while its cost scales with
`K`: 1.4x the rows at `K=1`, 5.7x at `K=6`.

It also moves a result: at `L=3` the leaf carry is what puts D-GRS ahead of RMC
(2.04x vs 1.92x with it, 1.90x vs 1.92x without — RMC is unaffected either way).
Without leaves, D-GRS's advantage comes from depth instead, and appears from
`L >= 4`.

### `--prefetch` — which drift the next round reuses

This matters more than anything else in this file. Every mode is **exact**; the
proposal only decides which states get drafted, and the verifier guarantees the
committed state is a target draw however stale the drift is. They trade
acceptance rate, never correctness.

- `nearest` (default) — the freshest drift available *at the committed step*:
  the committed leaf's own if `--evaluate-leaves` (exact), else the nearest
  drafted sibling at that depth, else the parent. Costs no extra evaluation.
- `parent` — the last verified parent's drift. Wrong on two axes at once: the
  wrong state *and* the previous step's noise level. specdiff's historical
  behaviour.
- `none` — reuse nothing, re-evaluate the target at every round's root. The
  best possible proposal, at **+1 NFE per round** — which more than eats the
  gain (1.11x against `nearest`'s 1.77x).

### `--evaluate-leaves` — what goes in the batch

Off by default. It only has an effect under `--prefetch nearest`, where it makes
the full-acceptance case *exact*: the committed leaf is the next root, so its
own drift is carried with no staleness at all. Under `parent` and `none` nothing
consumes leaf drifts and evaluating them is pure waste.

The cost is `K^L` extra rows per round — `verification_budget` goes from `B / K`
to `B + 1`. Measured at `L=3`: 1.89x → 2.02x at `K=4`, for 270 → 1000 target
rows per trajectory. It also moves a result; see Leaves below.

When it fires, one further row is saved: the committed leaf's target mean *is*
the next root's, so Phase 2 drops the root from its batch. That is the only case
where skipping the root is provably exact rather than an approximation — and it
is checked under `check_contract=True`, which re-derives the root mean and
asserts it matches.

### `--match` — how the RMC chain is sized

- `verification` (default) — equal **target** batch: `chain(|I|)`. The
  hardware-matched arm, since `|I|` is what has to fit in memory.
- `budget` — equal **proposal** budget: `chain(B)`. The paper's protocol, and
  what produced the committed `results/figure3`.

Both clamp to the horizon, which is why they mostly coincide: once the matched
depth exceeds `N` they both give `chain(N)`. Measured difference at most 0.13x.

At `d=512, eps=0.06` this is the difference between **1.27x and 1.77x**. The
default matches the reference implementation run with `--no-leaf-carry`,
validated per cell:

| rule | K | specdiff `nearest` | reference |
| --- | --- | --- | --- |
| d-grs | 1 | 1.77x | 1.77x |
| d-grs | 2 | 1.83x | 1.85x |
| d-grs | 4 | 1.89x | 1.90x |
| rmc | 2 | 1.93x | 1.93x |
| rmc | 4 | 1.91x | 1.92x |

`rmc K=1` is the one cell that differs (1.78x vs 1.87x): the reference's
`reflection_round` extends its block by one node regardless of `--no-leaf-carry`
(that is a separate flag, `carry_root_velocity`), so it keeps a leaf-equivalent.
At `chain(3)` that single row is +33% of the batch; by `K=4` it is +3.6%, which
is why the gap closes. Excluding leaves for both rules is the consistent rule.

### Deterministic steps

Two of the `T` steps carry zero transition noise — the first (`sigma = 1`) and
the last (`sigma_next = 0`) — where proposal and target are distinct point
masses and no coupling is possible. `NoiseSchedule` refuses a non-positive scale
(Remark 3), so `models.build()` exposes the `T - 2` speculative steps and
reports the two it dropped.

Speedups are therefore quoted as `num_steps / target_calls` — the speculative
steps against the calls that produced them, one baseline call per step. Using
the full `T` in the numerator would credit speculation for endpoints it never
simulated and inflate every cell by ~7%. Validated against the reference
implementation, which runs all 30: the two deterministic steps cost it ~1.0
extra call (measured 0.84–1.29 across cells; the first is free because the
warm-up makes the frozen drift exact at `Y_0`, the last forces a rejection), so
its `30 / calls` and this `28 / calls` agree to under 1%.

### Seeds

Three independent concerns, three seeds. The mixture layout is drawn once from
`--mixture-seed` and never varied. Per-trajectory streams are keyed by
`(K, L, replicate)` via `SeedSequence(spawn_key=...)` — indexable rather than
sequential, so re-running one cell reproduces it exactly regardless of what ran
before. The key is shared by both rules (pairing the comparison on the initial
state) and varies across cells (so each cell is an independent estimate and a
flat panel shows its own Monte-Carlo floor, ~±0.01x at 100 trajectories).
