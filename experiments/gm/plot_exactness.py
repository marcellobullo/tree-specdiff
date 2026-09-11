"""The exactness figure: speculation and refinement leave the sampled law alone.

    python experiments/gm/plot_exactness.py --run results/gm-exactness --eps 0.1

Needs a run whose replicates are independent *within a cell* (any picard_sweep
run is) and is read one arm at a time -- see exactness.py for why arms are never
pooled into a single test, and why the reference is plain target sampling rather
than the analytic mixture. Writes <run>/figures/exactness_eps<E>.{png,pdf}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm import models                                             # noqa: E402
from gm.exactness import (assign, energy_distance, mixture_components,  # noqa: E402
                          target_samples, whiten, whitening_frame)

# dataviz reference palette, light surface: slots 1-3 (all-pairs validated).
RMC, DGRS, REF = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, INK_3, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#dedcd6"
SEQ = ["#fbfcfe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f0a58c", "#e34948", "#a52322"]


def load_arms(run: Path, eps: float):
    """One entry per (rule, K, L, J) cell: replicates within a cell are i.i.d."""
    arms = {}
    cells = run / f"eps{eps:g}"
    # K*_L*/<rule>/J*, or K*_L*/J* holding every rule in runs before schema 4.
    paths = [*cells.glob("K*_L*/*/J*/samples.npz"), *cells.glob("K*_L*/J*/samples.npz")]
    for path in sorted(paths):
        K, L = map(int, re.search(r"/K(\d+)_L(\d+)/", str(path)).groups())
        J = int(re.search(r"/J(\d+)/", str(path)).group(1))
        z = np.load(path)
        for rule in np.unique(z["rule"]):
            arms[(str(rule), K, L, J)] = z["sample"][z["rule"] == rule]
    if not arms:
        raise SystemExit(f"no samples under {run}/eps{eps:g}")
    return arms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("results/gm-exactness"))
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--reference", type=int, default=60000, help="plain-target draws")
    p.add_argument("--null-replicates", type=int, default=200)
    p.add_argument("--perturbations", type=float, nargs="+", default=[1.02, 1.03, 1.05],
                   help="width errors that calibrate what the test can resolve")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--min-count", type=int, default=20,
                   help="reference counts a difference bin needs before it is drawn")
    p.add_argument("--recompute", action="store_true",
                   help="redo the energy-distance tests instead of reusing the cache")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    d, k = cfg["dimension"], cfg["num_components"]
    setting = models.build(dimension=d, num_components=k, num_steps=cfg["num_steps"],
                           eps=args.eps, mixture_seed=cfg["mixture_seed"])
    means, sds = mixture_components(d, k, cfg["mixture_seed"])
    arms = load_arms(args.run, args.eps)
    n_arm = min(len(v) for v in arms.values())
    print(f"arms: {len(arms)}, {n_arm} independent replicates each")

    cache = args.run / f"eps{args.eps:g}" / "_target_reference.npy"
    if cache.exists():
        reference = np.load(cache)
        print(f"reference: reusing {cache} ({len(reference)} draws)")
    else:
        print(f"reference: sampling {args.reference} plain-target draws ...", flush=True)
        rng = np.random.default_rng(args.seed)
        reference = target_samples(setting, rng.standard_normal((args.reference, d)), rng)
        np.save(cache, reference)

    # Coordinates: whiten by component off the reference, then a fixed 2-frame.
    ref_label = assign(reference, means)
    centres, scales, frame = whitening_frame(reference, ref_label, k, args.seed + 1)
    print("within-component sd  sampler:", np.round(scales, 4),
          " analytic:", np.round(sds, 4), " ratio:", np.round(scales / sds, 3))
    # Two coordinates off the same whitening: the fixed 2-frame is what the eye
    # reads, the full 512 is what the test uses. A 2-D projection discards 510
    # directions and with them the power to see anything but a gross error --
    # measured, not assumed: it cannot separate a 5% width error from the null.
    labels = {key: assign(v, means) for key, v in arms.items()}
    v_ref = whiten(reference, ref_label, centres, scales)
    v_arm = {key: whiten(x, labels[key], centres, scales) for key, x in arms.items()}
    u_ref = v_ref @ frame
    u_arm = {key: value @ frame for key, value in v_arm.items()}

    occ_ref = np.bincount(ref_label, minlength=k) / len(reference)
    occ_arm = {key: np.bincount(lab, minlength=k) / len(lab) for key, lab in labels.items()}

    # Null: the reference against itself, at the arms' own sample size. The same
    # draw with a deliberate width error calibrates what the test can see --
    # without it "no evidence of a difference" carries no scale.
    stats_cache = args.run / f"eps{args.eps:g}" / "_exactness_stats.npz"
    if stats_cache.exists() and not args.recompute:
        blob = np.load(stats_cache, allow_pickle=True)
        null, power = blob["null"], blob["power"].item()
        stats = blob["stats"].item()
        print(f"statistics: reusing {stats_cache}")
        render(args, cfg, means, reference, arms, u_ref, u_arm, occ_ref, occ_arm,
               stats, null, power, n_arm)
        return
    rs = np.random.default_rng(args.seed + 2)
    null = np.empty(args.null_replicates)
    power = {factor: np.empty(args.null_replicates) for factor in args.perturbations}
    for i in range(args.null_replicates):
        pick = rs.permutation(len(v_ref))[:2 * n_arm]
        a, b = v_ref[pick[:n_arm]], v_ref[pick[n_arm:]]
        null[i] = energy_distance(a, b)
        for factor in args.perturbations:
            power[factor][i] = energy_distance(a * factor, b)
    stats = {}
    for key, u in v_arm.items():
        pick = rs.permutation(len(v_ref))[:n_arm]
        value = energy_distance(u[:n_arm], v_ref[pick])
        stats[key] = (value, float((null >= value).mean()))
        print(f"  {key[0]:6s} K={key[1]} L={key[2]} J={key[3]}  "
              f"energy {value:+.5f}  p={stats[key][1]:.3f}   "
              f"occupancy {np.round(occ_arm[key], 3)}")
    print(f"null: median {np.median(null):+.5f}, 95th {np.quantile(null, .95):+.5f}")
    for factor, values in power.items():
        print(f"a {100 * (factor - 1):.0f}% width error would read "
              f"{np.median(values):+.5f} (p={(null >= np.median(values)).mean():.3f})")
    np.savez(stats_cache, null=null, power=power, stats=stats)

    render(args, cfg, means, reference, arms, u_ref, u_arm, occ_ref, occ_arm,
           stats, null, power, n_arm)


def render(args, cfg, means, reference, arms, u_ref, u_arm, occ_ref, occ_arm,
           stats, null, power, n_arm):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    div = LinearSegmentedColormap.from_list("div", DIV)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_3,
        "ytick.color": INK_3, "axes.edgecolor": RULE, "axes.linewidth": .8,
        "font.size": 9, "axes.titlesize": 9.5, "legend.frameon": False,
        "xtick.major.size": 3, "ytick.major.size": 3, "axes.titlepad": 21,
    })
    fig = plt.figure(figsize=(14.2, 7.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.42, 1, 1, 1], height_ratios=[1, .8],
                          hspace=.58, wspace=.34, left=.045, right=.955,
                          top=.855, bottom=.085)

    def titled(ax, tag, title, subtitle):
        ax.set_title(f"{tag}   {title}", loc="left", color=INK, fontweight="medium")
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=7.8,
                color=INK_2, va="bottom")

    # (a) the target, in the plane of its own component means -----------------
    ax = fig.add_subplot(gs[:, 0])
    centred = means - means.mean(0)
    # The five means span four directions, so most 2-planes superimpose a pair.
    # Take the pair of principal directions that keeps them furthest apart.
    directions = np.linalg.svd(centred, full_matrices=False)[2][:4].T
    def separation(pair):
        flat = centred @ directions[:, list(pair)]
        gaps = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
        return gaps[np.triu_indices(len(flat), 1)].min()
    basis = directions[:, list(max(itertools.combinations(range(directions.shape[1]), 2),
                                   key=separation))]
    zoom = 12.0                      # modes sit ~250x further apart than they are wide
    label_of = lambda x: np.argmin(((x[:, None, :] - means[None]) ** 2).sum(-1), 1)
    project = lambda x, lab: (((x - means[lab]) * zoom + means[lab]) - means.mean(0)) @ basis
    rs = np.random.default_rng(0)
    show = min(len(reference), min(len(v) for v in arms.values()) * 2)
    pick = rs.permutation(len(reference))[:show]
    ax.scatter(*project(reference[pick], label_of(reference[pick])).T, s=5, c=REF,
               alpha=.16, lw=0, label=f"plain target  ({show:,} shown)")
    for rule, colour in (("rmc", RMC), ("d-grs", DGRS)):
        pooled = np.concatenate([v for key, v in arms.items() if key[0] == rule])
        pooled = pooled[rs.permutation(len(pooled))[:show]]
        ax.scatter(*project(pooled, label_of(pooled)).T, s=4, c=colour, alpha=.16, lw=0,
                   label=f"{rule}  ({len(pooled):,} shown)")
    table = ["mode   target      arms"]
    for j in range(len(means)):
        share = [occ_arm[key][j] for key in sorted(occ_arm)]
        table.append(f"  {j + 1}    {occ_ref[j]:.3f}   {min(share):.3f}–{max(share):.3f}")
    ax.text(.025, .975, "\n".join(table), transform=ax.transAxes, va="top", ha="left",
            fontsize=7.4, color=INK_2, family="DejaVu Sans Mono", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=RULE, lw=.7))
    span = np.ptp(ax.get_ylim())
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + .22 * span)   # room for the table
    titled(ax, "a", "The target law: five modes in 512 dimensions",
           f"every mode holds its share of the mass   ·   spread magnified ×{zoom:g}")
    ax.set_xlabel("mean-plane direction 1"); ax.set_ylabel("mean-plane direction 2")
    ax.legend(loc="lower right", fontsize=8, markerscale=3.2, labelspacing=.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # (b, c) one whitened view, drawn for the reference and for the arms ------
    def contour_levels(density, masses=(.5, .9)):
        """Levels enclosing the given share of the reference mass."""
        flat = np.sort(density.ravel())[::-1]
        share = np.cumsum(flat) / flat.sum()
        return sorted(float(flat[np.searchsorted(share, m)]) for m in masses)

    lim, nbin = 3.6, 36
    bins = np.linspace(-lim, lim, nbin + 1)
    pooled_arm = np.concatenate(list(u_arm.values()))
    h_ref = np.histogram2d(*u_ref.T, bins=bins)[0]
    h_arm = np.histogram2d(*pooled_arm.T, bins=bins)[0]
    p_ref, p_arm = h_ref / h_ref.sum(), h_arm / h_arm.sum()
    top = max(p_ref.max(), p_arm.max())
    for col, (dens, tag, title, n) in enumerate((
            (p_ref, "b", "Plain target", len(u_ref)),
            (p_arm, "c", "Every arm, pooled", len(pooled_arm)))):
        axd = fig.add_subplot(gs[0, col + 1])
        axd.pcolormesh(bins, bins, dens.T, cmap=seq, vmin=0, vmax=top, rasterized=True)
        mid = (bins[:-1] + bins[1:]) / 2
        axd.contour(mid, mid, p_ref.T, levels=contour_levels(p_ref), colors=INK_2,
                    linewidths=.7, alpha=.55)
        axd.set_aspect("equal"); axd.set_xlabel("whitened $u_1$")
        titled(axd, tag, title, f"{n:,} samples, identical bins and colour scale")
        if col == 0:
            axd.set_ylabel("whitened $u_2$")

    # (d) their difference, in standard errors --------------------------------
    axz = fig.add_subplot(gs[0, 3])
    var = np.maximum(p_ref, 1e-12) * (1 - p_ref)
    se = np.sqrt(var / n_arm + var / len(u_ref))   # n_arm, not the pooled count:
    # every arm replays one replicate's initial noise, so the six are paired,
    # not independent. One arm's worth of error is the conservative choice.
    z = np.where(h_ref >= args.min_count, (p_arm - p_ref) / se, np.nan)
    mesh = axz.pcolormesh(bins, bins, z.T, cmap=div, norm=TwoSlopeNorm(0, -4, 4),
                          rasterized=True)
    axz.set_aspect("equal"); axz.set_xlabel("whitened $u_1$")
    titled(axz, "d", "Difference (c − b)",
           f"max |z| = {np.nanmax(np.abs(z)):.1f} over {int(np.isfinite(z).sum())} bins "
           f"(≥ {args.min_count} draws)")
    bar = fig.colorbar(mesh, ax=axz, fraction=.045, pad=.03, ticks=[-4, -2, 0, 2, 4])
    bar.outline.set_edgecolor(RULE); bar.ax.tick_params(length=2)

    # (e) every arm against a null the reference calibrates itself ------------
    axe = fig.add_subplot(gs[1, 1:])
    lo, hi = np.quantile(null, [.025, .975])
    axe.axhspan(lo, hi, color="#eceae4", lw=0,
                label=f"plain target vs itself, 95% of {len(null)} splits")
    axe.axhline(np.median(null), color=INK_3, lw=1, ls=(0, (4, 3)), zorder=2)
    resolvable = [f for f in sorted(power) if np.median(power[f]) > hi]
    shown = resolvable[0] if resolvable else max(power)
    axe.axhline(np.median(power[shown]), color="#a52322", lw=1.3, ls=(0, (5, 2)), zorder=2,
                label=f"a {100 * (shown - 1):.0f}% error in the width reads here")
    for rule, colour, dy in (("rmc", RMC, 13), ("d-grs", DGRS, -19)):
        keys = sorted((key for key in stats if key[0] == rule), key=lambda t: t[3])
        axe.plot([key[3] for key in keys], [stats[key][0] for key in keys], "-o",
                 color=colour, lw=2, ms=8, mec=SURFACE, mew=1.4, label=rule, zorder=3)
        for key in keys:
            axe.annotate(f"p={stats[key][1]:.2f}", (key[3], stats[key][0]),
                         textcoords="offset points", xytext=(0, dy), ha="center",
                         fontsize=7.6, color=INK_2)   # ink, not the series colour:
            # the marker beside the number already carries identity
    biggest = max(power)
    axe.set_xlabel("Picard iterations $J$")
    axe.set_ylabel("energy distance\nto the target law")
    axe.set_xticks(sorted({key[3] for key in stats}))
    axe.set_xlim(-0.45, max(key[3] for key in stats) + 0.45)
    span = np.median(power[shown]) - min(lo, min(v for v, _ in stats.values()))
    axe.set_ylim(min(lo, min(v for v, _ in stats.values())) - .35 * span,
                 np.median(power[shown]) + .55 * span)
    titled(axe, "e", "Distance from the target law, one test per arm",
           f"{n_arm:,} independent replicates each   ·   "
           f"p = share of the {len(null)} null splits at least this far out")
    axe.annotate(f"a {100 * (biggest - 1):.0f}% error: "
                 f"{np.median(power[biggest]) / max(np.median(power[shown]), 1e-12):.0f}× "
                 f"higher still", (1.0, np.median(power[shown])),
                 xycoords=("axes fraction", "data"), xytext=(-4, 5),
                 textcoords="offset points", ha="right", fontsize=7.6, color=INK_2)
    axe.grid(axis="y", color="#efedE7", lw=.8); axe.set_axisbelow(True)
    axe.legend(fontsize=8, ncols=2, loc="upper left", columnspacing=1.4)
    for side in ("top", "right"):
        axe.spines[side].set_visible(False)

    cells = sorted({(key[1], key[2]) for key in stats})
    fig.suptitle("Speculation and Picard refinement leave the sampled law unchanged",
                 x=.045, y=.965, ha="left", fontsize=13.5, color=INK)
    fig.text(.045, .915, f"Gaussian mixture, d = {cfg['dimension']}, T = {cfg['num_steps']}, "
             f"ε = {args.eps:g}, " + ", ".join(f"K={K} L={L}" for K, L in cells)
             + f"; reference is plain target sampling on the same {cfg['num_steps'] - 2}-step chain",
             fontsize=9.5, color=INK_2, ha="left")
    out = args.out or args.run / "figures" / f"exactness_eps{args.eps:g}"
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(f"{out}{suffix}", dpi=200)
    print("wrote", f"{out}.png", "and", f"{out}.pdf")


if __name__ == "__main__":
    main()
