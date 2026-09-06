"""Regression tests for the normalized GM Picard experiment output."""

from __future__ import annotations

import csv

import numpy as np

from experiments.gm import models, picard_sweep, plot_picard


def _config():
    return {
        "seed": 123,
        "prefetch": "nearest",
        "evaluate_leaves": False,
        "check_contract": False,
    }


def test_recorded_picard_prefix_and_cost_decomposition():
    setting = models.build(
        dimension=6,
        num_components=3,
        num_steps=8,
        eps=0.06,
        mixture_seed=7,
    )
    built = picard_sweep.build_sampler(
        setting, "d-grs", 2, 3, 2, _config()
    )
    bundle = picard_sweep.one_trajectory(
        setting, "d-grs", 2, 3, 2, 0, _config(), built
    )
    trajectory = bundle["trajectory"]

    assert trajectory["target_calls"] == (
        trajectory["proposal_target_calls"]
        + trajectory["refinement_target_calls"]
        + trajectory["verification_target_calls"]
    )
    assert trajectory["target_states_evaluated"] == (
        trajectory["proposal_target_states_evaluated"]
        + trajectory["refinement_target_states_evaluated"]
        + trajectory["verification_target_states_evaluated"]
    )
    assert bundle["refinements"]
    assert all(
        row["accepted"] == 1 and row["delta"] == 0.0
        for row in bundle["levels"]
        if row["level"] <= 2
    )
    assert {
        row["refinement_iteration"] for row in bundle["refinements"]
    } == {1, 2}
    assert max(row["round_index"] for row in bundle["refinements"]) == (
        trajectory["rounds"] - 1
    )


def test_atomic_cells_consolidate_and_summarize(tmp_path):
    setting = models.build(
        dimension=4,
        num_components=2,
        num_steps=7,
        eps=0.06,
        mixture_seed=11,
    )
    cfg = _config()
    built = picard_sweep.build_sampler(setting, "rmc", 1, 2, 1, cfg)
    bundles = [
        picard_sweep.one_trajectory(
            setting, "rmc", 1, 2, 1, replicate, cfg, built
        )
        for replicate in range(2)
    ]

    cell = picard_sweep.write_cell(
        tmp_path, "rmc", 1, 2, 1, bundles, save_samples=True
    )
    picard_sweep.consolidate(tmp_path)

    assert (cell / "COMPLETE").exists()
    with np.load(cell / "samples.npz") as samples:
        assert samples["initial"].shape == (2, 4)
        assert samples["sample"].shape == (2, 4)
        assert samples["trajectory"].shape == (2, setting.num_steps + 1, 4)

    with (tmp_path / "trajectories.csv").open(newline="") as handle:
        trajectories = list(csv.DictReader(handle))
    with (tmp_path / "levels.csv").open(newline="") as handle:
        levels = list(csv.DictReader(handle))
    with (tmp_path / "refinements.csv").open(newline="") as handle:
        refinements = list(csv.DictReader(handle))

    assert len(trajectories) == 2
    summary = plot_picard.summarize_trajectories(trajectories)
    level_summary = plot_picard.summarize_levels(levels)
    refinement_summary = plot_picard.summarize_refinements(refinements)
    assert summary[0]["n"] == 2
    assert all(
        row["conditional_rejection_hazard"] == 0.0
        for row in level_summary
        if row["level"] <= row["J"]
    )
    assert all("rejection_hazard_ci95_low" in row for row in level_summary)
    assert all("delta_ci95_halfwidth" in row for row in level_summary)
    assert plot_picard._wilson_interval(0, 100)[1] > 0.0
    assert refinement_summary
    assert all("current_delta_se" in row for row in refinement_summary)
