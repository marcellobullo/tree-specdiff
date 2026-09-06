"""Summarize and plot the normalized output of picard_sweep.py.

The summary CSVs are always written. Matplotlib is needed only for figures.
One trade-off, depth-profile, convergence, and sample-norm figure is produced
for every (rule, K, L) family present in the input.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics
from pathlib import Path

import numpy as np

GROUP_FIELDS = ("rule", "K", "L", "J")
TRAJECTORY_METRICS = (
    "target_calls",
    "target_states_evaluated",
    "proposal_target_calls",
    "proposal_target_states_evaluated",
    "refinement_target_calls",
    "refinement_target_states_evaluated",
    "verification_target_calls",
    "verification_target_states_evaluated",
    "verification_target_means_reused",
    "rounds",
    "accepted_depth",
    "acceptance_rate",
    "mean_committed",
    "drafted_states",
    "speculative_speedup",
    "effective_total_target_calls",
    "end_to_end_speedup",
    "sample_l2",
    "sample_mean",
    "sample_std",
    "sampling_seconds",
)


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row, field):
    value = row.get(field, "")
    return math.nan if value in ("", None) else float(value)


def _key(row, fields=GROUP_FIELDS):
    return tuple(row[field] for field in fields)


def _groups(rows, fields=GROUP_FIELDS):
    out = collections.defaultdict(list)
    for row in rows:
        out[_key(row, fields)].append(row)
    return out


def _mean(values):
    finite = [x for x in values if math.isfinite(x)]
    return statistics.mean(finite) if finite else math.nan


def _std(values):
    finite = [x for x in values if math.isfinite(x)]
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def _se(values):
    finite = [x for x in values if math.isfinite(x)]
    return _std(finite) / math.sqrt(len(finite)) if finite else math.nan


def _percentile(values, q):
    finite = [x for x in values if math.isfinite(x)]
    return float(np.percentile(finite, q)) if finite else math.nan


def _wilson_interval(successes, total, z=1.96):
    """Wilson score interval, including meaningful bounds at 0% and 100%."""

    if not total:
        return math.nan, math.nan
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _trajectory_values(group, field):
    """Average repeated node/round observations within each trajectory first."""

    clustered = collections.defaultdict(list)
    for row in group:
        clustered[row["trajectory_id"]].append(_float(row, field))
    return [_mean(values) for values in clustered.values()]


def summarize_trajectories(rows):
    summary = []
    for (rule, K, L, J), group in sorted(_groups(rows).items()):
        row = {
            "rule": rule,
            "K": int(K),
            "L": int(L),
            "J": int(J),
            "n": len(group),
            "B": int(group[0]["B"]),
            "verification_budget": int(group[0]["verification_budget"]),
            "dimension": int(group[0]["dimension"]),
            "num_steps": int(group[0]["num_steps"]),
            "total_steps": int(group[0]["total_steps"]),
        }
        for metric in TRAJECTORY_METRICS:
            values = [_float(item, metric) for item in group]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _std(values)
            row[f"{metric}_se"] = _se(values)
        accepted = sum(_float(item, "accepted_depth") for item in group)
        reached = sum(_float(item, "verified_levels") for item in group)
        row["acceptance_rate_pooled"] = accepted / reached
        summary.append(row)
    return summary


def summarize_levels(rows):
    fields = GROUP_FIELDS + ("level",)
    summary = []
    for (rule, K, L, J, level), group in sorted(_groups(rows, fields).items()):
        accepted = sum(int(item["accepted"]) for item in group)
        rejected = len(group) - accepted
        hazard_by_trajectory = _trajectory_values(group, "rejected")
        hazard = _mean(hazard_by_trajectory)
        hazard_se = _se(hazard_by_trajectory)
        if _std(hazard_by_trajectory) == 0.0:
            # Normal intervals collapse at an observed boundary. Wilson keeps
            # a meaningful finite-sample bound, using trajectories as the
            # independent trials rather than correlated verification events.
            hazard_low, hazard_high = _wilson_interval(
                hazard * len(hazard_by_trajectory), len(hazard_by_trajectory)
            )
        else:
            hazard_low = max(0.0, hazard - 1.96 * hazard_se)
            hazard_high = min(1.0, hazard + 1.96 * hazard_se)
        deltas = _trajectory_values(group, "delta")
        mismatch = _trajectory_values(group, "mean_mismatch_l2")
        delta_se = _se(deltas)
        summary.append({
            "rule": rule,
            "K": int(K),
            "L": int(L),
            "J": int(J),
            "level": int(level),
            "reached": len(group),
            "n_trajectories": len(hazard_by_trajectory),
            "accepted": accepted,
            "rejected": rejected,
            "conditional_acceptance": 1.0 - hazard,
            "conditional_rejection_hazard": hazard,
            "conditional_rejection_hazard_pooled": rejected / len(group),
            "rejection_hazard_se": hazard_se,
            "rejection_hazard_ci95_low": hazard_low,
            "rejection_hazard_ci95_high": hazard_high,
            "delta_mean": _mean(deltas),
            "delta_std": _std(deltas),
            "delta_se": delta_se,
            "delta_ci95_halfwidth": 1.96 * delta_se,
            "delta_median": _percentile(deltas, 50),
            "delta_p90": _percentile(deltas, 90),
            "delta_max": max(deltas),
            "delta_zero_fraction": sum(value == 0.0 for value in deltas) / len(deltas),
            "mean_mismatch_l2_mean": _mean(mismatch),
        })
    return summary


def summarize_refinements(rows):
    fields = GROUP_FIELDS + ("refinement_iteration", "node_depth")
    summary = []
    for key, group in sorted(_groups(rows, fields).items()):
        rule, K, L, J, iteration, depth = key
        delta = _trajectory_values(group, "current_delta")
        change = _trajectory_values(group, "iterate_change_l2")
        change_rms = _trajectory_values(group, "iterate_change_rms")
        summary.append({
            "rule": rule,
            "K": int(K),
            "L": int(L),
            "J": int(J),
            "refinement_iteration": int(iteration),
            "node_depth": int(depth),
            "n": len(group),
            "n_trajectories": len(delta),
            "current_delta_mean": _mean(delta),
            "current_delta_std": _std(delta),
            "current_delta_se": _se(delta),
            "current_delta_median": _percentile(delta, 50),
            "current_delta_p90": _percentile(delta, 90),
            "current_delta_max": max(delta),
            "current_delta_zero_fraction": sum(x == 0.0 for x in delta) / len(delta),
            "iterate_change_l2_mean": _mean(change),
            "iterate_change_l2_std": _std(change),
            "iterate_change_l2_se": _se(change),
            "iterate_change_l2_median": _percentile(change, 50),
            "iterate_change_rms_mean": _mean(change_rms),
        })
    return summary


def summarize_rounds(rows):
    summary = []
    for (rule, K, L, J), group in sorted(_groups(rows).items()):
        row = {
            "rule": rule,
            "K": int(K),
            "L": int(L),
            "J": int(J),
            "n_rounds": len(group),
        }
        for metric in (
            "committed",
            "accepted_depth",
            "drafted",
            "verified",
            "proposal_target_calls",
            "refinement_target_calls",
            "verification_target_calls",
            "verification_target_means_reused",
            "target_calls",
            "target_states_evaluated",
        ):
            values = [_float(item, metric) for item in group]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _std(values)
        row["rejection_fraction"] = _mean([_float(item, "rejected") for item in group])
        summary.append(row)
    return summary


def _write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def _family_slug(rule, K, L):
    return f"{rule.replace('-', '_')}-K{K}-L{L}"


def _family_groups(rows):
    return _groups(rows, ("rule", "K", "L"))


def _save(fig, stem):
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", dpi=300)
    print(f"wrote {stem.with_suffix('.png')} and .pdf")


def _ci95(rows, metric):
    return np.array([1.96 * float(row[f"{metric}_se"]) for row in rows])


def plot_tradeoffs(summary, figures):
    import matplotlib.pyplot as plt

    for (rule, K, L), rows in sorted(_family_groups(summary).items()):
        rows = sorted(rows, key=lambda row: int(row["J"]))
        J = np.array([int(row["J"]) for row in rows])
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)

        ax = axes[0, 0]
        acceptance = np.array([float(r["acceptance_rate_mean"]) for r in rows])
        acceptance_error = _ci95(rows, "acceptance_rate")
        committed = np.array([float(r["mean_committed_mean"]) for r in rows])
        committed_error = _ci95(rows, "mean_committed")
        ax.plot(J, acceptance, "o-", color="tab:blue", label="acceptance")
        ax.fill_between(
            J,
            np.maximum(0.0, acceptance - acceptance_error),
            np.minimum(1.0, acceptance + acceptance_error),
            alpha=0.2,
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("mean accepted fraction", color="tab:blue")
        ax.tick_params(axis="y", colors="tab:blue")
        ax.spines["left"].set_color("tab:blue")
        ax2 = ax.twinx()
        ax2.plot(J, committed, "s--", color="tab:orange", label="steps / round")
        ax2.fill_between(
            J,
            committed - committed_error,
            committed + committed_error,
            color="tab:orange",
            alpha=0.15,
        )
        ax2.set_ylabel("mean committed steps / round", color="tab:orange")
        ax2.tick_params(axis="y", colors="tab:orange")
        ax2.spines["right"].set_color("tab:orange")
        handles = ax.get_lines() + ax2.get_lines()
        ax.legend(handles, [line.get_label() for line in handles], frameon=False)

        ax = axes[0, 1]
        speculative = np.array([
            float(r["speculative_speedup_mean"]) for r in rows
        ])
        end_to_end = np.array([
            float(r["end_to_end_speedup_mean"]) for r in rows
        ])
        ax.plot(J, speculative, "o-", label="speculative region")
        ax.fill_between(
            J,
            speculative - _ci95(rows, "speculative_speedup"),
            speculative + _ci95(rows, "speculative_speedup"),
            alpha=0.2,
        )
        ax.plot(J, end_to_end, "s--", label="full trajectory (+ endpoints)")
        ax.fill_between(
            J,
            end_to_end - _ci95(rows, "end_to_end_speedup"),
            end_to_end + _ci95(rows, "end_to_end_speedup"),
            color="tab:orange",
            alpha=0.15,
        )
        ax.axhline(1.0, color="0.4", linestyle=":")
        ax.set_ylabel("target-call speedup")
        ax.legend(frameon=False)

        ax = axes[1, 0]
        bottom = np.zeros(len(rows))
        for metric, label in (
            ("proposal_target_calls_mean", "proposal"),
            ("refinement_target_calls_mean", "refinement"),
            ("verification_target_calls_mean", "verification"),
        ):
            values = np.array([float(r[metric]) for r in rows])
            ax.bar(J, values, bottom=bottom, label=label)
            bottom += values
        ax.errorbar(
            J,
            bottom,
            yerr=_ci95(rows, "target_calls"),
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
            label="total 95% CI",
        )
        ax.set_ylabel("target calls / trajectory")
        ax.set_xlabel("Picard iterations $J$")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1, 1]
        bottom = np.zeros(len(rows))
        for metric, label in (
            ("proposal_target_states_evaluated_mean", "proposal"),
            ("refinement_target_states_evaluated_mean", "refinement"),
            ("verification_target_states_evaluated_mean", "verification"),
        ):
            values = np.array([float(r[metric]) for r in rows])
            ax.bar(J, values, bottom=bottom, label=label)
            bottom += values
        ax.errorbar(
            J,
            bottom,
            yerr=_ci95(rows, "target_states_evaluated"),
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
            label="total 95% CI",
        )
        reused = [float(r["verification_target_means_reused_mean"]) for r in rows]
        ax.errorbar(
            J,
            reused,
            yerr=_ci95(rows, "verification_target_means_reused"),
            fmt="ko--",
            ms=3,
            capsize=2,
            label="reused verification means",
        )
        ax.set_ylabel("target rows / trajectory")
        ax.set_xlabel("Picard iterations $J$")
        ax.legend(frameon=False, fontsize=8)

        for ax in axes.flat:
            ax.grid(alpha=0.25)
            ax.set_xticks(J)
        fig.suptitle(f"{rule}   K={K}, L={L}   (bands: 95% CI)")
        _save(fig, figures / f"{_family_slug(rule, K, L)}-tradeoff")
        plt.close(fig)


def _matrix(rows, value):
    js = sorted({int(row["J"]) for row in rows})
    levels = sorted({int(row["level"]) for row in rows})
    lookup = {(int(row["J"]), int(row["level"])): float(row[value]) for row in rows}
    array = np.full((len(js), len(levels)), np.nan)
    for i, J in enumerate(js):
        for j, level in enumerate(levels):
            array[i, j] = lookup.get((J, level), np.nan)
    return js, levels, array


def _annotate_heatmap(
    ax,
    values,
    formatter,
    color_values=None,
    interval_low=None,
    interval_high=None,
    exact=None,
):
    colors = values if color_values is None else color_values
    image = ax.images[-1]
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                red, green, blue, _ = image.cmap(image.norm(colors[i, j]))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "black" if luminance > 0.5 else "white"
                text = formatter(values[i, j])
                if exact is not None and exact[i, j]:
                    text += "\nexact"
                elif interval_low is not None and interval_high is not None:
                    text += (
                        f"\n95% CI [{formatter(interval_low[i, j])},"
                        f"{formatter(interval_high[i, j])}]"
                    )
                ax.text(
                    j, i, text, ha="center", va="center",
                    fontsize=7, color=color, linespacing=0.9,
                )


def plot_depth_profiles(level_summary, figures):
    import matplotlib.pyplot as plt

    for (rule, K, L), rows in sorted(_family_groups(level_summary).items()):
        js, levels, hazard = _matrix(rows, "conditional_rejection_hazard")
        _, _, hazard_low = _matrix(rows, "rejection_hazard_ci95_low")
        _, _, hazard_high = _matrix(rows, "rejection_hazard_ci95_high")
        _, _, delta = _matrix(rows, "delta_mean")
        _, _, delta_half = _matrix(rows, "delta_ci95_halfwidth")
        exact = np.array([
            [level <= J for level in levels]
            for J in js
        ])
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)

        image = axes[0].imshow(hazard, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
        _annotate_heatmap(
            axes[0],
            hazard,
            lambda x: f"{x:.2f}",
            interval_low=hazard_low,
            interval_high=hazard_high,
            exact=exact,
        )
        axes[0].set_title(
            r"$\Pr(\mathrm{reject\ at\ level\ }\ell\mid"
            r"\mathrm{level\ }\ell\ \mathrm{was\ reached})$"
        )
        fig.colorbar(image, ax=axes[0], fraction=0.046)

        positive = delta[np.isfinite(delta) & (delta > 0)]
        floor = max(float(positive.min()) * 0.1, 1e-15) if len(positive) else 1e-15
        shown = np.log10(np.maximum(delta, floor))
        image = axes[1].imshow(shown, aspect="auto", cmap="viridis")
        _annotate_heatmap(
            axes[1],
            delta,
            lambda x: f"{x:.2g}",
            color_values=shown,
            interval_low=np.maximum(0.0, delta - delta_half),
            interval_high=delta + delta_half,
            exact=exact,
        )
        axes[1].set_title(
            r"$\delta_u^{(J)}="
            r"\frac{\left\|m_s^q(X_u^{(J)})-\widetilde{\mu}_u^{(J)}\right\|_2}"
            r"{\sigma_s}$"
        )
        bar = fig.colorbar(image, ax=axes[1], fraction=0.046)
        bar.set_label(r"$\log_{10}(\mathrm{mean}\ \delta_u^{(J)})$")

        for ax in axes:
            ax.set_xticks(range(len(levels)), levels)
            ax.set_yticks(range(len(js)), js)
            ax.set_xlabel(r"verification level $\ell\in\{1,\ldots,L\}$")
            ax.set_ylabel("Picard iterations $J$")
        fig.suptitle(f"{rule}   K={K}, L={L}")
        _save(fig, figures / f"{_family_slug(rule, K, L)}-depth")
        plt.close(fig)


def plot_convergence(refinement_summary, figures):
    import matplotlib.pyplot as plt

    for (rule, K, L), rows in sorted(_family_groups(refinement_summary).items()):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
        plotted_J = max(int(row["J"]) for row in rows)
        rows = [row for row in rows if int(row["J"]) == plotted_J]
        depths = sorted({int(row["node_depth"]) for row in rows})
        colors = plt.get_cmap("tab10")
        for color_index, depth in enumerate(depths):
            selected = sorted(
                (row for row in rows if int(row["node_depth"]) == depth),
                key=lambda row: int(row["refinement_iteration"]),
            )
            iterations = np.array([
                int(row["refinement_iteration"]) for row in selected
            ])
            delta = np.array([
                float(row["current_delta_mean"]) for row in selected
            ])
            delta_error = np.array([
                1.96 * float(row["current_delta_se"]) for row in selected
            ])
            change = np.array([
                float(row["iterate_change_l2_mean"]) for row in selected
            ])
            change_error = np.array([
                1.96 * float(row["iterate_change_l2_se"]) for row in selected
            ])
            color = colors(color_index % 10)
            axes[0].plot(
                iterations, np.maximum(delta, 1e-15), "o-",
                color=color, label=f"depth {depth}",
            )
            axes[0].fill_between(
                iterations,
                np.maximum(delta - delta_error, 1e-15),
                np.maximum(delta + delta_error, 1e-15),
                color=color,
                alpha=0.2,
            )
            axes[1].plot(
                iterations, np.maximum(change, 1e-15), "o-",
                color=color, label=f"depth {depth}",
            )
            axes[1].fill_between(
                iterations,
                np.maximum(change - change_error, 1e-15),
                np.maximum(change + change_error, 1e-15),
                color=color,
                alpha=0.2,
            )
        axes[0].set_yscale("log")
        axes[0].set_ylabel(
            r"$\mathbb{E}_{\mathrm{trajectories,nodes}}[\delta_u^{(j-1)}]$"
        )
        axes[0].set_title("proposal-target mismatch")
        axes[1].set_yscale("log")
        axes[1].set_ylabel(
            r"$\mathbb{E}[\|X_u^{(j-1)}-X_u^{(j-2)}\|_2]$"
        )
        axes[1].set_title("Picard iterate convergence")
        for ax in axes:
            ax.set_xlabel("Picard iteration $j$")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"{rule}   K={K}, L={L}, J={plotted_J}   (bands: 95% CI)"
        )
        _save(fig, figures / f"{_family_slug(rule, K, L)}-convergence")
        plt.close(fig)


def plot_sample_norms(out, trajectory_summary, figures):
    import matplotlib.pyplot as plt

    for (rule, K, L), rows in sorted(_family_groups(trajectory_summary).items()):
        data, labels = [], []
        for row in sorted(rows, key=lambda item: int(item["J"])):
            J = int(row["J"])
            path = out / "cells" / f"{rule.replace('-', '_')}-K{K}-L{L}-J{J}" / "samples.npz"
            if not path.exists():
                continue
            with np.load(path) as archive:
                sample = archive["sample"]
            data.append(np.linalg.norm(sample.reshape(len(sample), -1), axis=1)
                        / math.sqrt(sample[0].size))
            labels.append(str(J))
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
        ax.boxplot(data, showfliers=False)
        positions = np.arange(1, len(labels) + 1)
        means = np.array([values.mean() for values in data])
        errors = np.array([
            1.96 * values.std(ddof=1) / math.sqrt(len(values))
            if len(values) > 1 else 0.0
            for values in data
        ])
        ax.errorbar(
            positions,
            means,
            yerr=errors,
            fmt="D",
            color="tab:red",
            capsize=3,
            ms=4,
            label="mean ± 95% CI",
        )
        ax.set_xticks(positions, labels)
        ax.set_xlabel("Picard iterations $J$")
        ax.set_ylabel(r"terminal $\|Y_T\|_2/\sqrt{d}$")
        ax.set_title(f"sampling-law diagnostic: {rule}, K={K}, L={L}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, figures / f"{_family_slug(rule, K, L)}-sample-norm")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/gm-picard")
    args = parser.parse_args(argv)
    out = Path(args.out)

    trajectories = _read(out / "trajectories.csv")
    rounds = _read(out / "rounds.csv")
    levels = _read(out / "levels.csv")
    refinements = _read(out / "refinements.csv")
    trajectory_summary = summarize_trajectories(trajectories)
    round_summary = summarize_rounds(rounds)
    level_summary = summarize_levels(levels)
    refinement_summary = summarize_refinements(refinements)
    _write(out / "summary.csv", trajectory_summary)
    _write(out / "round_summary.csv", round_summary)
    _write(out / "level_summary.csv", level_summary)
    _write(out / "refinement_summary.csv", refinement_summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        figures = out / "figures"
        figures.mkdir(exist_ok=True)
        plot_tradeoffs(trajectory_summary, figures)
        plot_depth_profiles(level_summary, figures)
        if refinement_summary:
            plot_convergence(refinement_summary, figures)
        plot_sample_norms(out, trajectory_summary, figures)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(f"cannot plot: {missing} is not installed; summary CSVs were written")


if __name__ == "__main__":
    main()
