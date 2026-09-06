# Gaussian mixture — replicating Figures 1 and 3

This experiment reproduces the Section 5.1 `(K, L)` sweep for both verifiers on an analytic
Gaussian-mixture target. It runs with NumPy and requires neither a GPU nor a neural network,
making it suitable for validating an installation before running image models.

| file | |
| --- | --- |
| `models.py` | the mixture target and its churn schedule |
| `gm_sweep.py` | the sweep — writes `raw.csv`, resumable per cell |
| `lazy.py` | cost simulator: realises only the committed branch, so the top of the grid is runnable |
| `validate_lazy.py` | gate — does `lazy.py` agree with the eager sampler? |
| `plot_gm.py` | `raw.csv` → `summary.csv` + both paper figures |
| `picard_sweep.py` | eager `(K,L,J)` refinement sweep with normalized diagnostic tables |
| `plot_picard.py` | refinement convergence, depth, cost, reuse, and sample diagnostics |
| `heatmap.py` | the annotated `(K, L)` grid panels |

The sweep itself needs only NumPy. **Plotting needs matplotlib, pandas and seaborn**,
none of which are core dependencies:

```bash
pip install -e '.[dev]'      # or '.[plots]' for plotting alone
```

## 1. Check the simulator first

```bash
python experiments/gm/validate_lazy.py
```

Expect `worst |z| = ... over 36 configurations; 0 failing` (it fails at
`|z| > 4`). The prefetch policy is implemented twice — once in the sampler,
once in `lazy.py` — so this is what catches the two drifting apart. Run it
before any sweep that uses `--lazy`.

## 2. Smoke test

```bash
python experiments/gm/gm_sweep.py --out /tmp/gm-smoke --dimension 64 --num-steps 12 --K-values 1 2 3 --L-values 1 2 --replicates 5 --n-workers 2
```

This short run validates the complete experiment path before starting the full sweep.

## 3. The full sweep

```bash
RUN=results/gm/$(date +%Y%m%d-%H%M%S)
python experiments/gm/gm_sweep.py --out $RUN --seed 14 --n-workers 8 --lazy
python experiments/gm/plot_gm.py  --out $RUN
```

Only those three flags are non-default. Everything else — `d=512`, 5
components, `T=30`, `eps=0.06`, `K, L ∈ 1..7`, 100 replicates,
`--prefetch nearest`, `--match verification`,
`--max-verification-budget 60000` — is already the default.

`--n-workers` is purely speed: streams are keyed by `(K, L, replicate)`, not by
evaluation order, so results are identical at any worker count. The sweep skips
cells already in `raw.csv`, so an interrupted run resumes.

**`summary.csv` is written by `plot_gm.py`, not by the sweep.** Both steps are
needed. If `plot_gm.py` prints `cannot plot: <module> is not installed`, the
`summary.csv` is still valid — install the named package and re-run the plot
step alone; the sweep does not need repeating.

## Outputs

| file | |
| --- | --- |
| `raw.csv` | one row per trajectory |
| `summary.csv` | one row per `(rule, K, L)` cell |
| `figure1_frontier.{pdf,png}` | speed-up against the verification budget — the efficiency frontier |
| `figure3_grid_speedup.{pdf,png}` | the `(K, L)` grid, speed-up per cell, one panel per rule |
| `figure3_grid_calls.{pdf,png}` | the same grid in raw target calls |
| `speedup_vs_k.{pdf,png}`, `calls_vs_budget.png`, `speedup_vs_budget.png` | supporting plots |

## Reproducibility settings

**The seed.** `--seed 14` is what `results/gm/20260820-193412` used; the CLI
default is `20260714`. Both produce valid runs, but `--seed 14` is required to reproduce that
directory cell by cell.

**`--match` decides what the RMC arm is.** The default, `verification`, gives
both rules the same *target* batch `|I|` — the hardware-matched comparison.
`--match budget` gives them the same *proposal* budget `B`, the paper's
protocol, and a chain `K` times longer. At `d=512, eps=0.06` that is the
difference between **1.27x and 1.77x**, so it is not a detail. It is recorded in
`config.json` for every run.

For details on budgets, leaf evaluation, prefetch policies, and deterministic endpoints, see
[`../README.md`](../README.md).

## Picard refinement experiments

Picard refinement requires the complete frozen draft tree and therefore uses
the eager sampler. It is deliberately separate from the lazy paper sweep.

Start with a small smoke run:

```bash
RUN=/tmp/gm-picard-smoke
python experiments/gm/picard_sweep.py --out "$RUN" \
  --dimension 32 --num-steps 10 --K-values 1 2 --L-values 3 \
  --J-values 0 1 2 3 --replicates 4
python experiments/gm/plot_picard.py --out "$RUN"
```

The recommended first full diagnostic holds the topology at `L=4`, uses
common random numbers across every refinement count, and includes both an RMC
chain and branching D-GRS trees:

```bash
RUN=results/gm-picard/$(date +%Y%m%d-%H%M%S)
python experiments/gm/picard_sweep.py --out "$RUN" \
  --seed 14 --K-values 1 2 3 --L-values 4 \
  --J-values 0 1 2 3 4 --replicates 100 --n-workers 8
python experiments/gm/plot_picard.py --out "$RUN"
```

RMC is run only for `K=1`; incompatible `K` values are skipped. Each
`(rule,K,L,J)` cell is saved atomically, and rerunning the same command
resumes completed cells.

The top-level output is normalized rather than restricted to one predetermined
figure:

| output | granularity and purpose |
| --- | --- |
| `trajectories.csv` | aggregate acceptance, speedup, target calls/rows by phase, reuse, and state summaries |
| `rounds.csv` | every sampler `RoundRecord`, including the complete cost decomposition |
| `levels.csv` | every verification event, including `delta`, drift geometry, candidates, and outcome |
| `refinements.csv` | every internal node at every sweep, including iterate change and current mismatch |
| `cells/*/samples.npz` | full initial, terminal, and committed trajectory arrays |
| `schema.json` | machine-readable column inventory |

`plot_picard.py` derives `summary.csv`, `round_summary.csv`,
`level_summary.csv`, and `refinement_summary.csv`, then writes one set of
figures per `(rule,K,L)`. The raw normalized tables remain the source of
truth, so new aggregations and plots do not require another sampling run.
Line and convergence panels show 95% confidence bands. Levelwise intervals
aggregate repeated node observations within each trajectory before estimating
uncertainty, so the independent unit remains the sampled trajectory. Heatmap
cells in the theorem-guaranteed prefix are labeled `exact` instead of showing
an empirical interval.

To isolate exact final-target reuse, compare `J=L` with and without
`--evaluate-leaves`. Without leaf evaluation, converged internal-node target
means can remove the final verification call entirely; with leaf evaluation,
the leaves still require a fresh final call.
