"""Generate both paper figures from ``gm_sweep.py`` output.

    python experiments/plot_gm.py --out results/gm/<run>

Both figures use the same ``raw.csv -> summary.csv`` reduction. The script
writes ``summary.csv`` and:

`figure3_grid_speedup.{pdf,png}` / `figure3_grid_calls.{pdf,png}`
    The `(K, L)` grid, one panel per rule on a shared colour scale, plus a
    signed difference panel. Drawn by `heatmap.py`.

`figure1_frontier.{pdf,png}`
    Speed-up against the **verification budget** `B_ver`, which is the quantity the two
    arms are matched on under `--match verification`: `|I| = B / K` target rows
    per round, not the proposal budget `B`. Drafting is a vector add under the
    delayed drift; `|I|` is what the hardware has to hold, so it is the fair
    x-axis for comparing compute budgets.

    Each curve is an efficiency frontier -- the best speed-up reachable at a
    budget of at most `|I|` -- which removes the aliasing that makes a pooled
    curve zig-zag, since several `(K, L)` pairs can share a budget and a
    deep-narrow tree behaves nothing like a wide-shallow one.

    The RMC curve flattens because a chain truncates to `min(depth, N - n)`:
    past `|I| = N` there is no more trajectory to look ahead into, so the extra
    budget cannot be used.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path

RULE_LABEL = {"rmc": "RMC (De Bortoli et al.)", "d-grs": "D-GRS (ours)"}
RULE_SHORT = {"rmc": "RMC", "d-grs": "D-GRS"}
RULE_COLOR = {"rmc": "#4C72B0", "d-grs": "#55A868"}
RULE_ORDER = ("rmc", "d-grs")


def internal_nodes(K: int, L: int) -> int:
    """`|I|` of a uniform (K, L) tree: the budget the cell was allocated.

    Not the same as the row's `verification_budget`, which for the RMC arm is
    the chain's *actual* batch after truncation to the horizon. Plotting the
    allocated budget keeps the two arms comparable even when RMC cannot use all
    of it.
    """
    return L if K == 1 else (K**L - 1) // (K - 1)


def load(path: Path):
    cells = collections.defaultdict(list)
    for row in csv.DictReader(path.open()):
        cells[(row["rule"], int(row["K"]), int(row["L"]))].append(row)
    return cells


def summarise(cells):
    out = []
    for (rule, K, L), rows in sorted(cells.items()):
        calls = [int(r["target_calls"]) for r in rows]
        # Like for like: `target_calls` covers the speculative steps only, so
        # the baseline is those same steps -- one target call each.
        steps = int(rows[0]["num_steps"])
        speed = [steps / c for c in calls]
        sd = st.stdev(speed) if len(speed) > 1 else 0.0
        out.append({
            "rule": rule, "K": K, "L": L, "B": int(rows[0]["B"]),
            "num_steps": steps,
            "verification_budget": int(rows[0]["verification_budget"]),
            "allocated_budget": internal_nodes(K, L),
            "n": len(rows),
            "calls_mean": st.mean(calls),
            "calls_std": st.stdev(calls) if len(calls) > 1 else 0.0,
            "calls_se": (st.stdev(calls) / len(calls) ** 0.5) if len(calls) > 1 else 0.0,
            "speedup_mean": st.mean(speed),
            "speedup_std": sd,
            "speedup_se": sd / len(speed) ** 0.5 if speed else 0.0,
            "rows_mean": st.mean(int(r["target_rows"]) for r in rows),
            "accept": st.mean(int(r["accepted_depth"]) for r in rows)
            / st.mean(int(r["rounds"]) for r in rows),
        })
    return out


def _style():
    import matplotlib as mpl
    return mpl.rc_context({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "font.size": 11, "axes.labelsize": 12, "axes.linewidth": 0.8,
        "figure.dpi": 200,
    })


def plot_heatmaps(summary, out_dir):
    """The paper figures, via the notebook's `heatmap_grid` (see heatmap.py).

    Two grids, each two panels side by side: speed-up per cell, and the same
    grid in raw target calls, both with the Monte-Carlo standard error. `drop_l1`
    removes the degenerate `L = 1` row -- no lookahead, so every rule pays one
    call per step and the row would otherwise eat the whole colour range.
    """
    import pandas as pd

    from heatmap import heatmap_grid

    df = pd.DataFrame(summary)
    labels = {"rmc": "RMC", "d-grs": "D-GRS (ours)"}

    heatmap_grid(
        df, rules=("rmc", "d-grs"),
        metric="speedup", band="se", band_fmt="{:.2f}",
        drop_l1=True,
        show_budget=True,
        titles=labels, panel_size=(3.1, 3.2), cbar="shared",
        highlight_best=False,
        save=out_dir / "figure3_grid_speedup", formats=("pdf", "png"), dpi=400,
    )
    heatmap_grid(
        df, rules=("rmc", "d-grs"),
        metric="calls", band="se", band_fmt="{:.2f}",
        drop_l1=True, titles=labels, highlight_best=False,
        cbar_extend="neither", robust=True,
        save=out_dir / "figure3_grid_calls", formats=("pdf", "png"), dpi=400,
    )


def plot_frontier(summary, out_stem, band="std"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def frontier(rule):
        """Best speed-up at an allocated budget of at most `|I|`."""
        d = sorted((s for s in summary if s["rule"] == rule),
                   key=lambda s: s["allocated_budget"])
        best, best_band, rows = -np.inf, 0.0, []
        for budget, group in _groupby(d, "allocated_budget"):
            top = max(group, key=lambda s: s["speedup_mean"])
            if top["speedup_mean"] > best:
                best, best_band = top["speedup_mean"], top[f"speedup_{band}"]
            rows.append((budget, best, best_band))
        return np.array(rows, float)

    rules = [r for r in RULE_ORDER if any(s["rule"] == r for s in summary)]
    with _style():
        fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
        ax.axhline(1.0, color="0.4", ls=(0, (5, 2)), lw=1.1)
        ax.text(1.05, 1.015, "standard sampler", ha="left", va="bottom",
                color="0.4", fontsize=8.5)

        handles = []
        for rule in rules:
            fr = frontier(rule)
            budget, speed, err = fr[:, 0], fr[:, 1], fr[:, 2]
            color = RULE_COLOR[rule]
            ax.fill_between(budget, speed - err, speed + err, color=color,
                            alpha=0.18, lw=0)
            ln, = ax.plot(budget, speed, "-", color=color, lw=2.0,
                          label=f"{RULE_LABEL[rule]}  —  {speed[-1]:.2f}$\\times$")
            ax.plot(budget, speed, "o", color=color, ms=3.0, mec="white", mew=0.4)
            handles.append(ln)

        all_speed = [s["speedup_mean"] for s in summary]
        ax.set_xscale("log")
        ax.set_xlim(0.85, max(s["allocated_budget"] for s in summary) * 1.6)
        ax.set_ylim(min(all_speed) - 0.16, max(all_speed) + 0.10)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
        ax.set_xlabel(r"verification budget  $B_{\rm ver} = |\mathcal{I}| = B/K$")
        ax.set_ylabel("speed-up over standard sampler")
        ax.set_title("Exact target sampling at a fraction of the cost",
                     fontsize=12.5, fontweight="bold", pad=8)
        ax.legend(handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.42),
                  frameon=False, fontsize=9.0, handlelength=1.6, labelspacing=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="0.88", lw=0.5)
        ax.set_axisbelow(True)
        fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {out_stem.with_suffix('.png')} and .pdf")


def _groupby(rows, key):
    out = collections.OrderedDict()
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out.items()


# --------------------------------------------------------------------------- #
# Exploratory views (the notebook's non-paper figures).
#
# These keep the notebook's working style -- default font, light grid -- rather
# than the serif treatment the paper figures use, so it stays obvious which is
# which at a glance.
# --------------------------------------------------------------------------- #
EXPLORATORY_RC = {"figure.dpi": 200, "axes.grid": True, "grid.alpha": 0.3}


def _pool(rows, value, spread):
    """Combine replicate statistics of configurations sharing a budget.

    Two `(K, L)` pairs can land on the same budget, so plotting against it needs
    their replicates pooled rather than averaged: an `n`-weighted mean, and a
    variance that adds the within-configuration and between-configuration parts.
    Averaging the means instead would understate the spread wherever a budget is
    shared by configurations that behave differently -- which is exactly where
    it is shared, since a deep-narrow tree is not a wide-shallow one.
    """
    import numpy as np

    n = np.array([r["n"] for r in rows], float)
    m = np.array([r[value] for r in rows], float)
    s = np.array([r[spread] for r in rows], float)
    total = n.sum()
    mean = (n * m).sum() / total
    var = ((n - 1) * s**2).sum() + (n * (m - mean) ** 2).sum()
    var = var / (total - 1) if total > 1 else 0.0
    std = float(np.sqrt(var))
    return mean, std, std / np.sqrt(total), int(total)


def _by_budget(summary, rule, value, spread):
    """Pooled `(budget, mean, std, se)` rows for one rule, sorted by budget."""
    import numpy as np

    groups = collections.OrderedDict()
    for r in sorted((s for s in summary if s["rule"] == rule),
                    key=lambda s: s["allocated_budget"]):
        groups.setdefault(r["allocated_budget"], []).append(r)
    out = [(b, *_pool(g, value, spread)[:3]) for b, g in groups.items()]
    return np.array(out, float)


def plot_calls_vs_budget(summary, out_stem, band="se"):
    """Mean target calls against `B_ver`; lower is better.

    Faint markers are the individual `(K, L)` configurations. They matter: at a
    given budget the tree behaves quite differently depending on *how* the
    budget is split between depth and width, so the pooled curve is not
    monotone. `plot_calls_by_depth` separates those out.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    baseline = float(summary[0]["num_steps"])
    with mpl.rc_context(EXPLORATORY_RC):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for rule in [r for r in RULE_ORDER if any(s["rule"] == r for s in summary)]:
            color = RULE_COLOR[rule]
            raw = [s for s in summary if s["rule"] == rule]
            ax.scatter([s["allocated_budget"] for s in raw],
                       [s["calls_mean"] for s in raw],
                       color=color, s=9, alpha=0.35, linewidth=0)
            d = _by_budget(summary, rule, "calls_mean", f"calls_{band}")
            b, mean, err = d[:, 0], d[:, 1], 1.96 * d[:, 2]
            ax.plot(b, mean, "o-", color=color, label=RULE_LABEL[rule],
                    markersize=4, linewidth=1.6)
            ax.fill_between(b, mean - err, mean + err, color=color, alpha=0.2,
                            linewidth=0)
        if baseline:
            ax.axhline(baseline, color="0.35", linestyle="--", linewidth=1.2,
                       label=f"standard sampler ({baseline:g} calls)")
        ax.set_xscale("log")
        ax.set_xlabel(r"verification budget  $B_{\rm ver} = |\mathcal{I}| = B/K$")
        ax.set_ylabel("target calls per trajectory")
        ax.set_title(f"Target calls vs. verification budget (band = 95% CI on the mean)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out_stem.with_suffix('.png')}")


def plot_speedup_vs_budget(summary, out_stem, band="se"):
    """The same data as a speed-up -- the number the paper reports."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    with mpl.rc_context(EXPLORATORY_RC):
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        for rule in [r for r in RULE_ORDER if any(s["rule"] == r for s in summary)]:
            d = _by_budget(summary, rule, "speedup_mean", f"speedup_{band}")
            b, mean, err = d[:, 0], d[:, 1], d[:, 2]
            ax.plot(b, mean, "o-", color=RULE_COLOR[rule], label=RULE_LABEL[rule],
                    markersize=4, linewidth=1.6)
            ax.fill_between(b, mean - err, mean + err, color=RULE_COLOR[rule],
                            alpha=0.2, linewidth=0)
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(r"verification budget  $B_{\rm ver} = |\mathcal{I}| = B/K$")
        ax.set_ylabel("speed-up over standard sampler")
        ax.set_title("NFE speed-up vs. verification budget")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out_stem.with_suffix('.png')}")


def plot_calls_by_depth(summary, out_stem, band="se"):
    """One line per lookahead `L`, with `K` increasing along it.

    Disentangles the two ways of spending a budget -- the pooled curves above
    mix them -- and shows where each rule saturates in each direction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    rules = [r for r in RULE_ORDER if any(s["rule"] == r for s in summary)]
    depths = sorted({s["L"] for s in summary})
    baseline = float(summary[0]["num_steps"])
    with mpl.rc_context(EXPLORATORY_RC):
        fig, axes = plt.subplots(1, len(rules), figsize=(4.2 * len(rules), 4.0),
                                 sharey=True, squeeze=False)
        cmap = plt.get_cmap("viridis")
        for ax, rule in zip(axes[0], rules):
            for depth in depths:
                dl = sorted((s for s in summary
                             if s["rule"] == rule and s["L"] == depth),
                            key=lambda s: s["allocated_budget"])
                if not dl:
                    continue
                frac = (depth - min(depths)) / max(1, max(depths) - min(depths))
                color = cmap(frac)
                b = [s["allocated_budget"] for s in dl]
                m = [s["calls_mean"] for s in dl]
                e = [s[f"calls_{band}"] for s in dl]
                ax.plot(b, m, "o-", color=color, markersize=4, linewidth=1.5,
                        label=f"L = {depth}")
                ax.fill_between(b, [x - y for x, y in zip(m, e)],
                                [x + y for x, y in zip(m, e)], color=color,
                                alpha=0.2, linewidth=0)
            ax.axhline(baseline, color="0.35", linestyle="--", linewidth=1.0)
            ax.set_xscale("log")
            ax.set_xlabel(r"$B_{\rm ver}$")
            ax.set_title(RULE_SHORT.get(rule, rule))
        axes[0][0].set_ylabel("target calls per trajectory")
        axes[0][-1].legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out_stem.with_suffix('.png')}")


def plot_speedup_vs_k(summary, out_stem, band="se", drop_l1=True):
    """Speed-up against the branching factor, one line per lookahead depth.

    The clearest statement of the topological argument, because it puts the two
    rules on the same axis and lets `K` do the talking:

    * **D-GRS** gets one line per `L`, rising with `K`. Width buys acceptance.
    * **RMC** builds no tree, so every `(K, L)` cell of its panel is really a
      chain of the matched budget -- they all collapse onto `K = 1`. It is drawn
      as a single marker at the median with a bar spanning the full range, which
      is the honest picture: RMC has no `K` axis to move along.

    `drop_l1` removes `L = 1`, where no rule has any lookahead to speculate with.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    rows = [s for s in summary if not (drop_l1 and s["L"] == 1)]
    rmc = [s["speedup_mean"] for s in rows if s["rule"] == "rmc"]
    tree = [s for s in rows if s["rule"] == "d-grs"]
    if not rmc or not tree:
        return
    lo, hi, mid = min(rmc), max(rmc), float(np.median(rmc))
    ceiling = hi

    # The quoted saturation value is the mean over the cells that actually
    # saturate -- those whose matched chain reaches the horizon. Taking it over
    # every cell would drag it down with small-budget configurations that have
    # not saturated at all, and hardcoding it would go stale the moment the
    # sweep is rerun.
    plateau = [s["speedup_mean"] for s in rows
               if s["rule"] == "rmc" and s["allocated_budget"] >= s["num_steps"]]
    saturation = float(np.mean(plateau)) if plateau else mid

    rc = {"font.family": "serif", "mathtext.fontset": "dejavuserif",
          "font.size": 10, "axes.labelsize": 11}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)

        # RMC: no tree, so every configuration collapses onto K = 1.
        ax.scatter([1], [mid], color="#ED6F55", s=20, zorder=10, marker="^",
                   linewidth=0.8)
        ax.errorbar([1], [mid], yerr=[[mid - lo], [hi - mid]], color="#ED6F55",
                    zorder=10, capsize=3, capthick=0.8, elinewidth=0.8)

        depths = sorted({s["L"] for s in tree})
        norm, cmap = Normalize(min(depths), max(depths)), plt.get_cmap("viridis")
        for L in depths:
            dl = sorted((s for s in tree if s["L"] == L), key=lambda s: s["K"])
            k = np.array([s["K"] for s in dl], float)
            m = np.array([s["speedup_mean"] for s in dl], float)
            e = np.array([s[f"speedup_{band}"] for s in dl], float)
            ax.fill_between(k, m - e, m + e, color=cmap(norm(L)), alpha=0.25,
                            linewidth=0, zorder=2)
            ax.plot(k, m, "o-", color=cmap(norm(L)), ms=3.4, lw=1.6, zorder=3)
            ax.annotate(f"$L={L}$", (k[-1], m[-1]), xytext=(6, 0),
                        textcoords="offset points", color=cmap(norm(L)),
                        fontsize=8, va="center")

        ax.annotate(
            rf"RMC saturates at $\approx {saturation:.1f}\times$",
            xy=(1, mid), xytext=(3.0, ceiling - 0.024), textcoords="data",
            fontsize=8.5, color="#ED6F55",
            arrowprops=dict(arrowstyle="->", color="#ED6F55", lw=0.5,
                            linestyle=":", shrinkA=0, shrinkB=0),
        )

        ys = np.array(rmc + [s["speedup_mean"] for s in tree], float)
        pad = 0.06 * (ys.max() - ys.min())
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        ks = sorted({s["K"] for s in rows})
        ax.set_xticks(ks)
        ax.set_xlim(min(ks) - 0.3, max(ks) + 0.9)
        ax.set_xlabel("Branching Factor $K$")
        ax.set_ylabel("Speedup Over Standard Sampling")
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          ticks=depths, pad=0.02, fraction=0.04)
        cb.set_label("Lookahead depth $L$ (D-GRS)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)

        for ext in ("pdf", "png"):
            fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=400)
        plt.close(fig)
    print(f"wrote {out_stem.with_suffix('.png')} and .pdf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figure3")
    p.add_argument("--band", default="std", choices=["std", "se"])
    args = p.parse_args()
    out = Path(args.out)

    summary = summarise(load(out / "raw.csv"))
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {out / 'summary.csv'}  ({len(summary)} cells)")

    try:
        plot_heatmaps(summary, out)
        plot_frontier(summary, out / "figure1_frontier", band=args.band)
        plot_calls_vs_budget(summary, out / "calls_vs_budget")
        plot_speedup_vs_budget(summary, out / "speedup_vs_budget")
        plot_calls_by_depth(summary, out / "calls_vs_budget_by_depth")
        plot_speedup_vs_k(summary, out / "speedup_vs_k")
    except ImportError as exc:
        # Report the dependency that raised the import error.
        missing = getattr(exc, "name", None) or str(exc)
        print(f"cannot plot: {missing} is not installed "
              f"({exc}). summary.csv was written; figures skipped.")
        print("  pip install -e '.[plots]'    # matplotlib, pandas, seaborn")


if __name__ == "__main__":
    main()
