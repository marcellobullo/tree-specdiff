# Gaussian mixture — replicating Figures 1 and 3

The Section 5.1 sweep: every `(K, L)` cell for both rules, on the analytic
Gaussian-mixture target. **No GPU and no network** — this is pure NumPy, and it
is the setting to check the sampler in before pointing it at a real model.

| file | |
| --- | --- |
| `models.py` | the mixture target and its churn schedule |
| `gm_sweep.py` | the sweep — writes `raw.csv`, resumable per cell |
| `lazy.py` | cost simulator: realises only the committed branch, so the top of the grid is runnable |
| `validate_lazy.py` | gate — does `lazy.py` agree with the eager sampler? |
| `plot_gm.py` | `raw.csv` → `summary.csv` + both paper figures |
| `heatmap.py` | the annotated `(K, L)` grid panels |

The sweep itself needs only NumPy. **Plotting needs matplotlib, pandas and seaborn**,
none of which are core dependencies:

```bash
pip install -e '.[plots]'
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

Seconds. Confirms the whole path runs before you commit hours to it.

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

## Two things to know

**The seed.** `--seed 14` is what `results/gm/20260820-193412` used; the CLI
default is `20260714`. Both are valid runs of the same experiment, but only
`--seed 14` reproduces that directory's numbers cell for cell rather than merely
in distribution.

**`--match` decides what the RMC arm is.** The default, `verification`, gives
both rules the same *target* batch `|I|` — the hardware-matched comparison.
`--match budget` gives them the same *proposal* budget `B`, the paper's
protocol, and a chain `K` times longer. At `d=512, eps=0.06` that is the
difference between **1.27x and 1.77x**, so it is not a detail. It is recorded in
`config.json` for every run.

For why any of this is set up the way it is — the two budgets, leaves, prefetch
policies, the deterministic endpoints — see [`../README.md`](../README.md).
