"""Figure: NFE speed-up against verification budget for the EDM image runs.

    python experiments/images/plot_edm.py --root results/edm --out results/edm/figures/speedup

Rows are datasets, columns the churn `eps`, and the two lines in a panel are
the two verification rules. Reads only the `meta.json` that `run_edm.py` writes
beside each cell's samples, so it runs anywhere -- no torch, no checkpoints.

Expected layout under `--root`, exactly as the sweep writes it:

    <root>/<dataset>_n<N>_eps<E>_<match>/K<K>_L<L>/<rule>/meta.json

The glob is that deep on purpose. `plain-target/` sits one level higher and is
skipped along with it: it is the 1.0x reference, not a point on either curve.
A variant sweep in a subdirectory (`results/edm/prefetch-parent/...`) is a
*different* sampler, not a repeat of the runs above it, so it never pools into
the same point -- plot it by pointing `--root` at that subdirectory.

x-axis
    The *allocated* verification budget `|I| = B / K` of the cell's uniform
    tree -- the rows the target evaluates per round, and the quantity the two
    rules are matched on under `--match verification`. It is the budget that
    costs something: under the delayed drift, drafting a state is a vector add
    and verifying one is a forward. The axis is labelled plain `B`, the paper's
    convention in a figure where the drafted-node count never appears;
    `--xlabel` overrides it, and `--x budget` plots that drafted count
    `B = K + ... + K^L` instead. Neither is read off the row's own
    `verification_budget`: for RMC that field is the chain's *actual* batch
    after truncation to the horizon, which is the allocation only while the
    chain still fits inside the remaining horizon -- it does here, but a longer
    chain would silently pull the two arms off a common x.

band
    `--band` chooses the measure -- standard error of the mean, or standard
    deviation -- and `--over` the population it runs over.

    `--over runs` (default) pools repeats of a cell: the same `(K, L)` under
    another seed, in a sibling run directory. It is zero-wide, and then not
    drawn, until the sweep is actually repeated. It is also the only honest
    band for `speedup` and `end_to_end_speedup`, which reduce a batch through
    a `max` over its trajectories and are one number per run.

    `--over images` uses the per-image NFE counts `r_i` that `run_edm.py`
    records in `metric_totals.target_calls_per_trajectory` (legacy files fall
    back to `rounds_per_trajectory`), so one run carries its
    own band. Defined for `mean_isolated_speedup` alone, the one reported
    metric that is a mean over images: the plotted mean is unchanged and only
    the band appears. Runs generated before that field existed have to be
    regenerated to get it.

metric
    `end_to_end_speedup` by default: the batch's NFE ratio across the whole
    sampler, the two deterministic Euler steps included -- what running the
    sampler actually costs. `--metric speedup` is the same ratio over the
    speculative steps alone. Both are batch numbers, held to the slowest
    trajectory in the batch; `--metric mean_isolated_speedup` is instead the
    mean over trajectories of the speed-up each would have reached alone, and
    sits above them.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import OrderedDict
from pathlib import Path

# The rules that get a line, in legend order, with the paper's names.
RULE_LABEL = OrderedDict([("d-grs", "D-GRS"), ("rmc", "RMC"), ("paws", "PAWS")])
RULE_COLOR = {"D-GRS": "#1D3557", "RMC": "seagreen", "PAWS": "#A85532"}
RULE_MARKER = {"D-GRS": "o", "RMC": "^", "PAWS": "s"}
RULE_LINESTYLE = {"D-GRS": "-", "RMC": "--", "PAWS": "-."}

DATASET_LABEL = {"cifar10": "CIFAR-10", "ffhq": "FFHQ"}
DATASET_ORDER = ("cifar10", "ffhq")

METRICS = ("end_to_end_speedup", "speedup", "mean_isolated_speedup")
# The per-image term behind a metric, where there is one: image `i` spent `r_i`
# target calls, so on its own it would have run `N / r_i` times faster. The two
# batch ratios have no such term -- see the `band` note above.
PER_IMAGE = {"mean_isolated_speedup": lambda steps, rounds: steps / max(rounds, 1)}
Y_LABEL = "NFE Speedup"
# `|I|` goes out as plain `B`: the paper's figures name the budget they plot,
# and the drafted count is not in them. Plotting both at once is what needs the
# two symbols kept apart.
X_LABEL = {"verification": r"Budget $B$", "budget": r"Draft Budget $B$",
           "k": r"Branching $K$"}

# `--x k` colours by lookahead instead of by rule, because the rule is already
# carried by the marker and the dash. Qualitative, not a ramp: `L` is a handful
# of small integers to tell apart, not a quantity to read off a gradient.
LOOKAHEAD_COLORS = ("#1D3557", "seagreen", "#C1666B", "#7D5BA6", "#B5651D",
                    "#2A9D8F", "#6C757D")

# `cifar10_n100_eps0.3_verification` -> dataset, sample count, churn, match.
RUN_DIR = re.compile(
    r"^(?P<dataset>.+?)_n(?P<n>\d+)_eps(?P<eps>[0-9.]+)(?:_(?P<match>.+))?$"
)


def proposal_budget(K: int, L: int) -> int:
    """`B`, the states a uniform `(K, L)` round drafts: `K + ... + K^L`."""
    return L if K == 1 else K * (K**L - 1) // (K - 1)


def verification_budget(K: int, L: int) -> int:
    """`|I|`, the internal nodes: the rows the target evaluates, `B / K`."""
    return L if K == 1 else (K**L - 1) // (K - 1)


def dataset_of(run_dir: str, meta: dict) -> str:
    """Dataset name from the run directory, falling back to the checkpoint.

    The directory is what the sweep names a run after; the checkpoint stem
    (`edm-cifar10-32x32-cond-vp`) is the same fact recorded independently, and
    covers a run directory named some other way.
    """
    matched = RUN_DIR.match(run_dir)
    if matched:
        return matched.group("dataset")
    stem = Path(str(meta.get("network", ""))).stem.split("-")
    return stem[1] if len(stem) > 1 and stem[0] == "edm" else (stem[0] or "unknown")


def load(root: Path, pattern: str = "*") -> list[dict]:
    """One record per `<run>/<cell>/<rule>/meta.json` under `root`."""
    rows, warned = [], []
    for path in sorted(root.glob(f"{pattern}/*/*/meta.json")):
        meta = json.loads(path.read_text())
        rule = meta.get("rule")
        if rule not in RULE_LABEL:  # plain-target and anything else unplotted
            continue
        K, L = int(meta["branching"]), int(meta["lookahead"])
        budget = proposal_budget(K, L)
        # The one place the derived budget can be checked against the run: the
        # tree arm drafts every node, so its own field must agree.
        requires_chain = meta.get("requires_chain", meta.get("chain_depth") is not None or rule == "rmc")
        if not requires_chain and int(meta.get("proposal_budget", budget)) != budget:
            warned.append(f"{path}: proposal_budget={meta['proposal_budget']}, "
                          f"(K={K}, L={L}) implies B={budget}")
        row = {
            "run": path.parents[2].name,
            "cell": path.parents[1].name,
            "dataset": dataset_of(path.parents[2].name, meta),
            "eps": float(meta["eps"]),
            "rule": rule,
            "method": RULE_LABEL[rule],
            "K": K,
            "L": L,
            # Lower-case duplicate of `K`: the `--x` choice, the row key and the
            # summary key are one name throughout, and the CLI choice is `k`.
            "k": K,
            "budget": budget,
            "verification": verification_budget(K, L),
            "seed": meta.get("seed"),
            "num_samples": meta.get("num_samples"),
            "match": meta.get("run_signature", {}).get("config", {}).get("match"),
            "prefetch": (meta.get("sampler") or {}).get("prefetch"),
            "acceptance_rate": meta.get("acceptance_rate"),
            "occupancy": meta.get("occupancy"),
            "speculative_steps": meta.get("speculative_steps"),
            "calls": list(meta.get("metric_totals", {}).get(
                "target_calls_per_trajectory", ()) or ()),
            "rounds": list(meta.get("metric_totals", {}).get(
                "rounds_per_trajectory", ()) or ()),
        }
        row.update({m: float(meta[m]) for m in METRICS if m in meta})
        rows.append(row)
    for line in warned:
        print(f"warning: {line}")
    return rows


def summarise(rows: list[dict], metric: str, x: str, over: str = "runs",
              split_lookahead: bool = False) -> list[dict]:
    """Pool runs into one point per `(dataset, eps, method, budget)`.

    `split_lookahead` adds `L` to that key, which is what `--x k` needs: two
    cells at one `K` and different `L` are the very thing that mode separates,
    so averaging them would erase the curve being asked for. On the budget axes
    the same collision is a deliberate pooling instead -- see below.

    Two things pool here. Repeats of a cell -- the same `(K, L)` under another
    seed -- are the case `--over runs` bands. Distinct cells landing on one
    budget are the other, and are worth knowing about: a deep-narrow tree and a
    wide-shallow one behave nothing alike at equal `B`, so the pooled point
    averages over that difference. `cells` records which ones went in.

    `over` names what the point's `n` counts and what the spread is taken over:
    the runs pooled here, or every image inside them. Under `--over images` the
    mean is recomputed from the per-image terms, which for
    `mean_isolated_speedup` reproduces the recorded value exactly -- pooling
    images across runs weights by image count, where pooling runs does not.
    """
    groups = OrderedDict()
    for row in rows:
        key = (row["dataset"], row["eps"], row["method"], row[x],
               row["L"] if split_lookahead else None)
        groups.setdefault(key, []).append(row)

    out = []
    for (dataset, eps, method, budget, lookahead), members in sorted(groups.items()):
        if over == "images":
            blank = [m for m in members if not (m.get("calls") or m["rounds"])]
            if blank:
                raise SystemExit(
                    f"{blank[0]['run']}/{blank[0]['cell']}/{blank[0]['rule']}: meta.json "
                    f"has no metric_totals.rounds_per_trajectory, so it carries no "
                    f"per-image spread. Regenerate the run, or use --over runs."
                )
            term = PER_IMAGE[metric]
            values = [term(m["speculative_steps"], r) for m in members for r in (m.get("calls") or m["rounds"])]
        else:
            values = [m[metric] for m in members]
        std = st.stdev(values) if len(values) > 1 else 0.0
        out.append({
            "dataset": dataset,
            "eps": eps,
            "method": method,
            x: budget,
            **({"lookahead": lookahead} if split_lookahead else {}),
            "over": over,
            "n": len(values),
            "mean": st.mean(values),
            "std": std,
            "sem": std / (len(values) ** 0.5),
            "ci": (1.96 * std) / (len(values) ** 0.5),
            "cells": "|".join(sorted({m["cell"] for m in members})),
            "runs": "|".join(sorted({m["run"] for m in members})),
        })
    return out


def _error_plot(x, y, err, color=None, **kwargs):
    """One rule's line in one panel, with its band. Mapped over the grid."""
    import matplotlib.pyplot as plt
    import pandas as pd

    label = kwargs.pop("label", None)
    marker = kwargs.pop("marker", "o")
    # Facets arrive in row order, not budget order; the line has to be drawn
    # left to right or it doubles back on itself.
    d = pd.DataFrame({"x": x, "y": y, "err": err}).sort_values("x")

    ax = plt.gca()
    # The label goes on the line alone -- a labelled band would show up in the
    # legend as a second entry for the same rule.
    ax.plot(d["x"], d["y"], color=color, marker=marker, label=label, **kwargs)
    if float(d["err"].abs().max()) > 0:
        ax.fill_between(d["x"], d["y"] - d["err"], d["y"] + d["err"],
                        color=color, alpha=0.2, linewidth=0)


def plot(summary: list[dict], out_stem: Path, *, metric: str, x: str, band: str,
         title: str, xlabel: str, height: float, aspect: float,
         split_lookahead: bool = False, formats=("pdf", "png")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", rc={
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "grid.color": "black",
        "grid.linewidth": 0.5,
    })

    df = pd.DataFrame(summary)
    df["err"] = 0.0 if band == "none" else df[band]
    # Facet on the formatted churn so the column titles read as written.
    df["eps_label"] = df["eps"].map(lambda e: f"{e:g}")

    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    datasets += sorted(set(df["dataset"]) - set(datasets))
    columns = [f"{e:g}" for e in sorted(set(df["eps"]))]
    methods = [m for m in RULE_LABEL.values() if m in set(df["method"])]

    if split_lookahead:
        # One line per (rule, L). Colour carries `L` and the marker and dash
        # carry the rule, so the rule stays readable exactly as it is on the
        # budget axes and `L` is the new thing the eye has to separate.
        lookaheads = sorted(set(df["lookahead"]))
        colour = {L: LOOKAHEAD_COLORS[i % len(LOOKAHEAD_COLORS)]
                  for i, L in enumerate(lookaheads)}
        df["series"] = [f"{m}, L={L:g}"
                        for m, L in zip(df["method"], df["lookahead"])]
        present = set(df["series"])
        # Rule-major, so one rule's lookaheads sit together in the legend.
        order = [f"{m}, L={L:g}" for m in methods for L in lookaheads
                 if f"{m}, L={L:g}" in present]
        by_series = dict(zip(df["series"], zip(df["method"], df["lookahead"])))
        hue, hue_order = "series", order
        palette = {s: colour[by_series[s][1]] for s in order}
        marker = [RULE_MARKER[by_series[s][0]] for s in order]
        linestyle = [RULE_LINESTYLE[by_series[s][0]] for s in order]
        legend_title = "Sampler and lookahead"
    else:
        hue, hue_order = "method", methods
        palette = {m: RULE_COLOR[m] for m in methods}
        marker = [RULE_MARKER[m] for m in methods]
        linestyle = [RULE_LINESTYLE[m] for m in methods]
        legend_title = "Sampler"

    g = sns.FacetGrid(
        data=df,
        row="dataset", row_order=datasets,
        col="eps_label", col_order=columns,
        hue=hue, hue_order=hue_order,
        palette=palette,
        height=height, aspect=aspect,
        sharey="row",  # speed-ups differ by dataset, not across a dataset's row
        hue_kws={"marker": marker, "linestyle": linestyle},
    )
    g.map(_error_plot, x, "mean", "err", linewidth=1.8)

    # Every panel draws the same two rules, so the per-panel handles are
    # duplicates of each other; a dict keyed by label keeps one of each.
    handles, labels = [], []
    for ax in g.axes.flat:
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        handles.extend(panel_handles)
        labels.extend(panel_labels)
    by_label = dict(zip(labels, handles))
    g.fig.legend(by_label.values(), by_label.keys(), title=legend_title,
                 bbox_to_anchor=(0.5, 1.02), loc="center",
                 ncol=len(by_label), frameon=False)

    g.set_axis_labels(xlabel or X_LABEL[x], Y_LABEL)
    for ax, dataset in zip(g.axes[:, 0], g.row_names):
        # The dataset names the row; it rides above the shared y-label rather
        # than on a right-hand row title, which the legend already crowds.
        ax.set_ylabel(f"{DATASET_LABEL.get(dataset, dataset)}\n{Y_LABEL}")
    g.set_titles(r"$\varepsilon={col_name}$")
    if title:
        plt.suptitle(title, fontsize=14, y=1.15, x=0.5, fontweight="bold")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        g.fig.savefig(out_stem.with_suffix(f".{ext}"), format=ext,
                      bbox_inches="tight", dpi=400)
    plt.close(g.fig)
    print(f"wrote {', '.join(str(out_stem.with_suffix('.' + e)) for e in formats)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default="results/edm", type=Path,
                   help="directory holding the run directories (default: results/edm)")
    p.add_argument("--runs", default="*",
                   help="glob selecting run directories under --root (default: *)")
    p.add_argument("--out", default="results/edm/figures/speedup_vs_budget", type=Path,
                   help="output stem; .pdf, .png and .csv are written")
    p.add_argument("--metric", default="mean_isolated_speedup", choices=METRICS)
    p.add_argument("--x", default="verification",
                   choices=("verification", "budget", "k"),
                   help="verification budget |I| (default), drafted budget B, or "
                        "the branching factor K. `k` draws one curve per "
                        "lookahead L, since cells sharing a K but not an L are "
                        "the comparison that mode exists to make and must not "
                        "be pooled")
    p.add_argument("--xlabel", default=None, help="override the x-axis label")
    p.add_argument("--band", default="sem", choices=("sem", "std", "ci", "none"),
                   help="shaded band: standard error, standard deviation, or none")
    p.add_argument("--over", default="runs", choices=("runs", "images"),
                   help="population the band is taken over (default: runs)")
    p.add_argument("--datasets", nargs="+", default=None, help="keep these datasets")
    p.add_argument("--eps", nargs="+", type=float, default=None, help="keep these churns")
    p.add_argument("--title", default="Pixel Space Diffusion",
                   help="figure title; empty string for none")
    p.add_argument("--height", type=float, default=2.0, help="panel height in inches")
    p.add_argument("--aspect", type=float, default=1.0, help="panel width / height")
    args = p.parse_args()

    rows = load(args.root, args.runs)
    if args.datasets:
        rows = [r for r in rows if r["dataset"] in set(args.datasets)]
    if args.eps:
        keep = {float(e) for e in args.eps}
        rows = [r for r in rows if r["eps"] in keep]
    if not rows:
        raise SystemExit(
            f"no runs under {args.root}/{args.runs}/*/*/meta.json after filtering; "
            f"expected <root>/<dataset>_n<N>_eps<E>_<match>/K<K>_L<L>/<rule>/meta.json"
        )
    missing = [r for r in rows if args.metric not in r]
    if missing:
        raise SystemExit(f"{missing[0]['run']}/{missing[0]['cell']}: "
                         f"meta.json has no {args.metric!r}")
    if args.over == "images" and args.metric not in PER_IMAGE:
        raise SystemExit(
            f"--over images needs a metric that is a mean over images; {args.metric} is "
            f"a ratio of batch totals, one number per run with no per-image term. Use "
            f"--metric {' or '.join(sorted(PER_IMAGE))}, or --over runs with repeated seeds."
        )

    split_lookahead = args.x == "k"
    summary = summarise(rows, args.metric, args.x, args.over, split_lookahead)

    csv_path = args.out.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {csv_path}  ({len(rows)} runs -> {len(summary)} points)")

    # Both are worth saying out loud: a flat band is a property of the sweep,
    # not of the rules, and a pooled point is not the cell it looks like.
    if args.band != "none" and all(s["n"] == 1 for s in summary):
        print("note: one run per point, so the band is zero-wide and not drawn; "
              "repeat the sweep under more seeds, or use --over images "
              f"with --metric {' / '.join(sorted(PER_IMAGE))}")
    for s in summary:
        if "|" in s["cells"]:
            print(f"note: {s['dataset']} eps={s['eps']:g} {s['method']} "
                  f"{args.x}={s[args.x]} pools {s['cells']}")

    try:
        plot(summary, args.out, metric=args.metric, x=args.x, band=args.band,
             title=args.title, xlabel=args.xlabel, height=args.height,
             aspect=args.aspect, split_lookahead=split_lookahead)
    except ImportError as exc:
        missing_module = getattr(exc, "name", None) or str(exc)
        print(f"cannot plot: {missing_module} is not installed ({exc}). "
              f"{csv_path} was written; figures skipped.")
        print("  pip install -e '.[plots]'    # matplotlib, pandas, seaborn")


if __name__ == "__main__":
    main()
