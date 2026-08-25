"""Annotated (K, L) heatmaps -- the paper figure.

Originally a port of a `heatmap_grid` helper written against the reference
implementation's sweep output; it now reads specdiff's own schema
(`rule`/`calls_mean` rather than `method`/`mean`) and its speed-up baseline.
See `plot_gm.py`.

One panel per rule on a **shared** colour scale, so cells are comparable across
panels: RMC builds no tree, so its `(L, K)` cell is a linear chain matched to
the tree cell at the same coordinates.

Cells are annotated with the **verification budget** `B_ver = |I| = B/K` -- the
target rows a round evaluates, and the quantity the two arms are matched on --
rather than the proposal budget `B`, which counts drafted states. Drafting is a
vector add under the delayed drift, so `B` overstates what the hardware sees by
a factor of `K`.

`B_ver` is the budget each cell is *allocated*. RMC cannot always spend it: its
chain truncates to `min(B_ver, N)`, which is why its panel flattens. The amount
actually evaluated is the `verification_budget` column of the summary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RULE_LABELS = {"rmc": "RMC", "d-grs": "D-GRS (ours)"}


def _compact(b: float) -> str:
    """1092 -> 1.09k, 137256 -> 137k (keeps the annotation two lines high)."""
    if not np.isfinite(b):
        return ""
    if b >= 1e6:
        return f"{b / 1e6:.3g}M"
    if b >= 1e3:
        return f"{b / 1e3:.3g}k"
    return f"{int(b)}"


def _strip_tex(s: str) -> str:
    """Rough printable length of an annotation line ($...$ and \\pm are narrow)."""
    return s.replace("$", "").replace("\\pm", "±").replace("\\times", "x")


def heatmap_grid(
    df,
    *,
    rules=("rmc", "d-grs"),
    metric="calls",                  # "calls" (lower better) | "speedup" (higher better)
    baseline=None,
    diff_panel=False,
    l_values=None, k_values=None,
    drop_l1=False,
    budget_col="allocated_budget",   # |I|, not the drafted-node count B
    annotate=True,
    value_fmt=None,
    band=None,                       # None | "std" | "se"
    band_fmt="{:.1f}",
    show_budget=True, budget_prefix="$B_{\\rm ver}$=",
    annot_size=None, annot_color=None, annot_kws=None,
    highlight_best=True, highlight_kw=None,
    cmap=None, diff_cmap="RdBu",
    shared_scale=True, vmin=None, vmax=None, robust=True,
    cbar="shared", cbar_extend=None, cbar_extendfrac=0.04,
    cbar_extendrect=False, cbar_label=None, cbar_shrink=0.92,
    edge_lw=0.6, edge_color="white", square=False, frame=False,
    panel_size=(3.1, 3.2), figsize=None,
    titles=None, title_size=10.0, label_size=9.0, tick_size=8.0,
    xlabel="$K$ (children per node)", ylabel="$L$ (lookahead)",
    suptitle=None, rc=None,
    save=None, formats=("pdf", "png"), dpi=400, transparent=False, show=False,
):
    """Returns ``(fig, axes)``. See the notebook cell this is ported from."""
    rules = [r for r in rules if r in set(df.rule)]
    if not rules:
        raise ValueError("none of `rules` is present in df.rule")
    speed = metric == "speedup"
    if value_fmt is None:
        value_fmt = "{:.2f}$\\times$" if speed else "{:.1f}"
    if cmap is None:
        cmap = "viridis" if speed else "viridis_r"   # yellow = fast either way
    # axes.grid off here as well as per-axes: a stray grid is the one thing that
    # cannot be undone after seaborn has drawn the mesh.
    rc = {"font.family": "serif", "mathtext.fontset": "dejavuserif",
          "axes.linewidth": 0.6, "axes.grid": False, **(rc or {})}

    def table(rule, field):
        t = df[df.rule == rule].pivot_table(index="L", columns="K", values=field)
        if l_values is not None:
            t = t.reindex(index=list(l_values))
        if k_values is not None:
            t = t.reindex(columns=list(k_values))
        if drop_l1 and 1 in t.index:
            t = t.drop(index=1)
        return t

    vals, bands = {}, {}
    for r in rules:
        if speed:
            mean = table(r, "speedup_mean")
            err = table(r, f"speedup_{band}") if band else None
        else:
            mean = table(r, "calls_mean")
            # calls +/- band, derived from the speed-up spread we recorded
            err = None
            if band:
                sp, sd = table(r, "speedup_mean"), table(r, f"speedup_{band}")
                err = mean * sd / sp
        vals[r], bands[r] = mean, err
    budgets = table(rules[0], budget_col)

    panels = list(rules)
    if diff_panel and len(rules) > 1:
        panels.append("__diff__")

    finite = np.concatenate([t.to_numpy(float).ravel() for t in vals.values()])
    finite = finite[np.isfinite(finite)]
    lo, hi = (np.percentile(finite, [2, 98]) if robust else (finite.min(), finite.max()))
    lo = lo if vmin is None else vmin
    hi = hi if vmax is None else vmax

    if figsize is None:
        figsize = (panel_size[0] * len(panels) + (0.55 if cbar == "shared" else 0.0),
                   panel_size[1])

    def panel_title(key, i):
        if isinstance(titles, dict) and key in titles:
            return titles[key]
        if titles is not None and not isinstance(titles, dict):
            return titles[i]
        names = {**RULE_LABELS, **(titles if isinstance(titles, dict) else {})}
        if key == "__diff__":
            a, b = names.get(rules[-1], rules[-1]), names.get(rules[0], rules[0])
            return f"{a} $-$ {b}" if speed else f"{b} $-$ {a}"
        return names.get(key, key)

    def cell_text(arr, band_arr):
        bud = budgets.to_numpy(float)
        out = np.full(arr.shape, "", dtype=object)
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if not np.isfinite(arr[r, c]):
                    continue
                s = value_fmt.format(arr[r, c])
                if band_arr is not None and np.isfinite(band_arr[r, c]):
                    s += "$\\pm$" + band_fmt.format(band_arr[r, c])
                if show_budget:
                    s += "\n" + budget_prefix + _compact(bud[r, c])
                out[r, c] = s
        return out

    with mpl.rc_context(rc):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False,
                                 constrained_layout=True)
        axes, meshes = axes[0], []
        for i, (ax, key) in enumerate(zip(axes, panels)):
            if key == "__diff__":
                # signed so that POSITIVE always means "rules[-1] is better"
                t = (vals[rules[-1]] - vals[rules[0]]) if speed else \
                    (vals[rules[0]] - vals[rules[-1]])
                arr = t.to_numpy(float)
                m = np.nanmax(np.abs(arr))
                kw = dict(cmap=diff_cmap, vmin=-m, vmax=m, center=0.0)
                band_arr = None
            else:
                t = vals[key]
                arr = t.to_numpy(float)
                kw = dict(cmap=cmap)
                if shared_scale:
                    kw.update(vmin=lo, vmax=hi)
                else:
                    p = np.percentile(arr[np.isfinite(arr)], [2, 98] if robust else [0, 100])
                    kw.update(vmin=p[0], vmax=p[1])
                band_arr = None if bands[key] is None else bands[key].to_numpy(float)

            texts = cell_text(arr, band_arr) if annotate else None
            size = annot_size
            if annotate and size is None:      # auto: widest line must fit one cell
                widest = max((max((len(_strip_tex(ln)) for ln in s.split("\n")), default=0)
                              for s in texts.ravel()), default=1)
                cell_pt = 72.0 * panel_size[0] * 0.78 / max(arr.shape[1], 1)
                size = float(np.clip(cell_pt / (0.63 * max(widest, 1)), 3.5, 9.0))
            akws = {"fontsize": size, "linespacing": 1.2, **(annot_kws or {})}
            if annot_color is not None:
                akws["color"] = annot_color

            sns.heatmap(t, ax=ax, annot=texts if annotate else None, fmt="",
                        annot_kws=akws, mask=~np.isfinite(arr),
                        linewidths=edge_lw, linecolor=edge_color, square=square,
                        cbar=False, **kw)
            meshes.append(ax.collections[0])

            # Cell edges are the mesh's own; no grid of any kind on top.
            ax.grid(False, which="both")
            ax.set_axisbelow(False)
            ax.set_title(panel_title(key, i), fontsize=title_size)
            ax.set_xlabel(xlabel, fontsize=label_size)
            ax.set_ylabel(ylabel if i == 0 else "", fontsize=label_size)
            ax.tick_params(labelsize=tick_size, length=0, labelleft=(i == 0))
            plt.setp(ax.get_yticklabels(), rotation=0)
            plt.setp(ax.get_xticklabels(), rotation=0)
            for sp in ax.spines.values():
                sp.set_visible(frame)

            if highlight_best and key != "__diff__":
                best = (np.nanargmax if speed else np.nanargmin)(arr)
                r, c = np.unravel_index(best, arr.shape)
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1, zorder=5,
                    **{"edgecolor": "crimson", "lw": 1.4, "fill": False,
                       **(highlight_kw or {})}))

        if cbar_label is None:
            cbar_label = ("speed-up over standard sampler" if speed
                          else "target calls per trajectory")
        extend = ("both" if robust else "neither") if cbar_extend is None else cbar_extend
        cbar_kw = dict(shrink=cbar_shrink, pad=0.02, extend=extend,
                       extendrect=cbar_extendrect)
        if cbar_extendfrac is not None:
            cbar_kw["extendfrac"] = cbar_extendfrac
        if cbar == "shared":
            cb = fig.colorbar(meshes[0], ax=list(axes), **cbar_kw)
            cb.set_label(cbar_label, fontsize=label_size)
            cb.ax.tick_params(labelsize=tick_size)
            cb.outline.set_linewidth(0.6)
        elif cbar == "each":
            for ax, mesh in zip(axes, meshes):
                cb = fig.colorbar(mesh, ax=ax, **cbar_kw)
                cb.ax.tick_params(labelsize=tick_size)
                cb.outline.set_linewidth(0.6)
        if suptitle:
            fig.suptitle(suptitle, fontsize=title_size + 1.5, fontweight="bold")

        if save is not None:
            for ext in formats:
                out = Path(save)
                out = out if out.suffix == f".{ext}" else out.with_suffix(f".{ext}")
                fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=transparent)
                print(f"wrote {out}")
        plt.show() if show else plt.close(fig)
    return fig, axes
