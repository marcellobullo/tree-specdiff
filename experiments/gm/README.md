# Gaussian mixture — replicating Figures 1 and 3

This experiment reproduces the Section 5.1 `(K, L)` sweep and adds PAWS on an analytic
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

The sweep itself needs NumPy and SciPy (installed with the core library). **Plotting needs matplotlib, pandas and seaborn**,
none of which are core dependencies:

```bash
pip install -e '.[dev]'      # or '.[plots]' for plotting alone
```

## 1. Check the simulator first

```bash
python experiments/gm/validate_lazy.py
```

Expect `worst |z| = ... over 60 configurations; 0 failing` (it fails at
`|z| > 4`). The prefetch policy is implemented twice — once in the sampler,
once in `lazy.py` — so this is what catches the two drifting apart. Run it
before any sweep that uses `--lazy`.

## 2. Smoke test

```bash
python experiments/gm/gm_sweep.py --out /tmp/gm-smoke --dimension 64 --num-steps 12 --K-values 1 2 3 --L-values 1 2 --replicates 5 --n-workers 2
```

This short run validates the complete experiment path before starting the full sweep.

The default rules are `d-grs rmc paws`; pass `--rules d-grs rmc` for the original
comparison. PAWS variants use `--verifier-options`, for example
`'{"paws":{"rank_policy":"max","residual_complement":"first"}}'`. This also works
with `--lazy` and with `picard_sweep.py`. Use a separate output directory for each
variant. See [PAWS](../../docs/paws.md) for the derivation and numerical caveats.

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
  --eps 0.1 --dimension 32 --num-steps 10 --K-values 1 2 --L-values 3 \
  --J-values 0 1 2 3 --replicates 4
python experiments/gm/plot_picard.py --out "$RUN/eps0.1"
```

The canonical sweep uses `K,L = 1,...,7`, 100 replicates, the epsilon grid
`0.1, 0.3, 0.6`, and the depth-specific Picard grid `J=0,...,L`. RMC is
matched separately to every `(K,L)` D-GRS tree. The initial protocol uses equal
verification batches and does not evaluate leaves:

```bash
RUN=results/gm-picard/$(date +%Y%m%d-%H%M%S)
python experiments/gm/picard_sweep.py --out "$RUN" \
  --eps-values 0.1 0.3 0.6 \
  --K-values 1 2 3 4 5 6 7 --L-values 1 2 3 4 5 6 7 \
  --J-up-to-L --rules rmc d-grs --replicates 100 \
  --match verification --no-evaluate-leaves --n-workers 1

for EPS in 0.1 0.3 0.6; do
  python experiments/gm/plot_picard.py --out "$RUN/eps$EPS"
done
```

The command-line defaults encode this epsilon, topology, replicate, matching,
and leaf-evaluation protocol; `--J-up-to-L` is explicit above to make the
per-depth refinement grid visible. `--J-up-to-L` and an explicit `--J-values`
list are mutually exclusive.

`--picard-update drift` (the default) freezes only the target's velocity during
refinement; `--picard-update increment` freezes the whole increment `m - x`, as
every run did before the flag existed. The choice is part of the saved protocol:
a `config.json` without it counts as `increment`, so older output directories
resume only with `--picard-update increment`, and mixing the two in one `--out`
directory is refused. See [docs/refinement.md](../../docs/refinement.md) for the
derivations and measured differences.

`--progress auto` (the default) displays one overall configuration bar in an
interactive terminal and periodic progress lines in redirected logs. The count
includes both sampler rules and advances for completed, resumed, and
budget-skipped configurations. Use `--progress bar`, `--progress plain`, or
`--progress none` to override the display mode.

Picard refinement is eager. In particular, `(K,L)=(7,7)` materializes almost a
million draft states, so the full grid requires a high-memory machine and should
start with one worker. `--max-verification-budget 60000` provides an optional
safety cap, but capped cells are deliberately absent from the output.

For each `(K,L)`, D-GRS uses `DraftTree.uniform(K,L)`. RMC uses a chain whose
depth matches that tree under `--match verification` (equal target batch) or
`--match budget` (equal proposal budget), clamped to the trajectory horizon.
`--evaluate-leaves` participates in verification matching as well as controlling
which rows the sampler evaluates.

Results have this hierarchy:

```text
RUN/
  config.json
  schema.json
  eps0.1/
    config.json
    trajectories.csv
    rounds.csv
    levels.csv
    refinement_summary.csv
    K2_L2/
      J0/
        trajectories.csv
        rounds.csv
        levels.csv
        refinement_summary.csv
        samples.npz
        COMPLETE
      J1/
      J2/
  eps0.3/
  eps0.6/
```

Each `J` directory contains both rules and is saved atomically. While it is
running, each replicate is checkpointed separately; rerunning the same command
resumes both completed directories and incomplete cells. Each epsilon directory
also gets normalized aggregate tables. Schema-v2 cells retain their raw
`refinements.csv` and receive a derived summary during resume; new schema-v3
cells contain only the summary:

To resume only part of an existing grid, pass subsets of its saved `--K-values`
and `--L-values`. For example, `--K-values 1 2 3 4 5 6 --L-values 1 2 3 4 5 6`
skips all cells with either `K=7` or `L=7`. Other protocol settings must still
match. The saved configuration retains the original grid so you can resume the
excluded cells later. Existing results are preserved, and the consolidated
epsilon tables still include all completed cells, including excluded ones.

| output | granularity and purpose |
| --- | --- |
| `trajectories.csv` | aggregate acceptance, speedup, allocated/actual budgets, target calls/rows by phase, reuse, and state summaries |
| `rounds.csv` | every sampler `RoundRecord`, including committed and accepted steps and the complete cost decomposition |
| `levels.csv` | every verification event, including `delta`, drift geometry, candidates, and outcome |
| `refinement_summary.csv` | online count, mean, standard deviation, RMS, extrema, percentiles, and zero fraction per trajectory/round/sweep/depth |
| `K*/J*/samples.npz` | rule labels plus full initial, terminal, and committed trajectory arrays |
| `schema.json` | machine-readable column inventory at the run root |

`plot_picard.py` derives `summary.csv`, `round_summary.csv`,
`level_summary.csv`, and `refinement_overview.csv`, then writes one set of
figures per `(rule,K,L)`. The normalized tables remain the source of truth, so
new aggregations and plots do not require another sampling run.
Line and convergence panels show 95% confidence bands. Levelwise intervals
aggregate repeated node observations within each trajectory before estimating
uncertainty, so the independent unit remains the sampled trajectory. Heatmap
cells in the theorem-guaranteed prefix are labeled `exact` instead of showing
an empirical interval.

To isolate exact final-target reuse, compare `J=L` with and without
`--evaluate-leaves`. Without leaf evaluation, converged internal-node target
means can remove the final verification call entirely; with leaf evaluation,
the leaves still require a fresh final call.
