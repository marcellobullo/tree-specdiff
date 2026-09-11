"""Regression tests for the normalized GM Picard experiment output."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from experiments.gm import models, picard_sweep, plot_picard


def _config():
    return {
        "seed": 123,
        "eps": 0.06,
        "match": "verification",
        "prefetch": "nearest",
        "evaluate_leaves": False,
        "check_contract": False,
    }


def test_j_up_to_l_builds_a_depth_specific_grid():
    args = picard_sweep.parser().parse_args(
        ["--L-values", "2", "4", "--J-up-to-L"]
    )

    assert list(picard_sweep.picard_iterations(args, 2)) == [0, 1, 2]
    assert list(picard_sweep.picard_iterations(args, 4)) == [0, 1, 2, 3, 4]


def test_canonical_defaults_and_default_j_grid():
    args = picard_sweep.parser().parse_args([])

    assert args.K_values == list(range(1, 8))
    assert args.L_values == list(range(1, 8))
    assert args.eps_values == [0.1, 0.3, 0.6]
    assert args.replicates == 100
    assert args.match == "verification"
    assert args.evaluate_leaves is False
    assert args.progress == "auto"
    assert list(picard_sweep.picard_iterations(args, 2)) == [0, 1, 2]
    assert "J_values" not in vars(args)
    assert "J_up_to_L" not in vars(args)


def test_configuration_progress_counts_the_complete_grid():
    args = picard_sweep.parser().parse_args([
        "--eps-values", "0.1", "0.3",
        "--K-values", "1", "2",
        "--L-values", "1", "2",
        "--J-up-to-L",
        "--rules", "rmc", "d-grs",
    ])

    assert picard_sweep.configuration_count(args) == 40

    stream = io.StringIO()
    progress = picard_sweep.ConfigurationProgress(4, mode="bar", stream=stream)
    progress.update(2, label="eps0.1/K1_L1/J0")
    progress.log("cell metrics")
    progress.update(2, label="complete", force=True)
    progress.close()
    output = stream.getvalue()
    assert "2/4 configurations" in output
    assert "4/4 configurations" in output
    assert "cell metrics" in output


def test_matched_chain_depth_uses_the_selected_budget():
    tree = picard_sweep.DraftTree.uniform(2, 3)

    assert picard_sweep.matched_chain_depth(tree, 30, "verification", False) == 7
    assert picard_sweep.matched_chain_depth(tree, 30, "budget", False) == 14
    assert picard_sweep.matched_chain_depth(tree, 5, "budget", False) == 5
    assert picard_sweep.matched_chain_depth(tree, 30, "verification", True) == 14


def test_j_up_to_l_is_mutually_exclusive_with_explicit_values():
    with pytest.raises(SystemExit):
        picard_sweep.parser().parse_args(
            ["--J-values", "0", "1", "--J-up-to-L"]
        )


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
    assert bundle["refinement_summary"]
    assert all(
        row["accepted"] == 1 and row["delta"] == 0.0
        for row in bundle["levels"]
        if row["level"] <= 2
    )
    assert {
        row["refinement_iteration"] for row in bundle["refinement_summary"]
    } == {1, 2}
    assert max(row["round_index"] for row in bundle["refinement_summary"]) == (
        trajectory["rounds"] - 1
    )
    assert all(row["node_count"] >= 1 for row in bundle["refinement_summary"])
    assert sum(
        row["node_count"] for row in bundle["refinement_summary"]
        if row["round_index"] == 0 and row["refinement_iteration"] == 1
    ) == built[3].verification_budget(evaluate_leaves=False)
    assert all(
        row["end_step"] == row["start_step"] + row["committed"]
        for row in bundle["rounds"]
    )
    assert bundle["rounds"][-1]["cumulative_committed"] == setting.num_steps
    assert bundle["rounds"][-1]["cumulative_target_calls"] == trajectory["target_calls"]
    assert bundle["rounds"][-1]["cumulative_target_states_evaluated"] == (
        trajectory["target_states_evaluated"]
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
    eps_out = tmp_path / picard_sweep.epsilon_slug(cfg["eps"])
    for rule in ("rmc", "d-grs"):
        built = picard_sweep.build_sampler(setting, rule, 2, 2, 1, cfg)
        bundles = [
            picard_sweep.one_trajectory(
                setting, rule, 2, 2, 1, replicate, cfg, built
            )
            for replicate in range(2)
        ]
        cell = picard_sweep.write_cell(
            eps_out, rule, 2, 2, 1, bundles, save_samples=True
        )

        assert cell == eps_out / "K2_L2" / rule / "J1"
        assert (cell / "COMPLETE").exists()
        with np.load(cell / "samples.npz") as samples:
            assert samples["initial"].shape == (2, 4)
            assert samples["sample"].shape == (2, 4)
            assert samples["trajectory"].shape == (2, setting.num_steps + 1, 4)
            assert samples["rule"].tolist() == [rule, rule]
            assert samples["eps"].tolist() == [cfg["eps"]] * 2
    picard_sweep.consolidate(eps_out)

    with (eps_out / "trajectories.csv").open(newline="") as handle:
        trajectories = list(csv.DictReader(handle))
    with (eps_out / "levels.csv").open(newline="") as handle:
        levels = list(csv.DictReader(handle))
    with (eps_out / "refinement_summary.csv").open(newline="") as handle:
        refinements = list(csv.DictReader(handle))

    assert len(trajectories) == 4
    assert {row["rule"] for row in trajectories} == {"rmc", "d-grs"}
    rmc = next(row for row in trajectories if row["rule"] == "rmc")
    dgrs = next(row for row in trajectories if row["rule"] == "d-grs")
    assert (rmc["B"], rmc["allocated_verification_budget"]) == ("6", "3")
    assert (rmc["actual_proposal_budget"], rmc["verification_budget"]) == ("3", "3")
    assert rmc["chain_depth"] == "3"
    assert (dgrs["actual_proposal_budget"], dgrs["verification_budget"]) == (
        "6", "3"
    )
    assert dgrs["chain_depth"] == ""
    summary = plot_picard.summarize_trajectories(trajectories)
    level_summary = plot_picard.summarize_levels(levels)
    refinement_summary = plot_picard.summarize_refinements(refinements)
    assert all(row["n"] == 2 for row in summary)
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


def test_replicate_checkpoints_resume_and_finalize(tmp_path):
    setting = models.build(
        dimension=4, num_components=2, num_steps=7, eps=0.06, mixture_seed=11
    )
    cfg = _config()
    destination, work = picard_sweep.streamed_cell_paths(
        tmp_path / "eps0.06", "d-grs", 2, 2, 1
    )
    assert destination == tmp_path / "eps0.06" / "K2_L2" / "d-grs" / "J1"
    built = picard_sweep.build_sampler(setting, "d-grs", 2, 2, 1, cfg)
    for replicate in range(2):
        bundle = picard_sweep.one_trajectory(
            setting, "d-grs", 2, 2, 1, replicate, cfg, built
        )
        picard_sweep.write_replicate(work, replicate, [bundle])

    assert picard_sweep.completed_replicates(work) == {0, 1}
    picard_sweep.finalize_streamed_cell(work, destination, 2)
    assert (destination / "COMPLETE").exists()
    assert not (destination / "replicates").exists()
    assert not (destination / "refinements.csv").exists()
    assert (destination / "refinement_summary.csv").exists()
    with np.load(destination / "samples.npz") as samples:
        assert samples["replicate"].tolist() == [0, 1]
        assert samples["rule"].tolist() == ["d-grs", "d-grs"]


def test_resume_subset_keeps_original_grid_and_selects_only_requested_cells(
    tmp_path, monkeypatch
):
    calls = []

    def record_run(args, out, cfg, eps, progress):
        calls.append((list(args.K_values), list(args.L_values), dict(cfg)))

    monkeypatch.setattr(picard_sweep, "_run_epsilon", record_run)
    base = ["--out", str(tmp_path), "--eps", "0.1", "--progress", "none"]
    picard_sweep.main(base + ["--J-up-to-L"])
    saved = (tmp_path / "config.json").read_text()
    picard_sweep.main(base + [
        "--J-up-to-L", "--K-values", "1", "2", "--L-values", "2", "3",
    ])
    assert calls[-1][:2] == ([1, 2], [2, 3])
    assert calls[-1][2]["K_values"] == list(range(1, 8))
    assert (tmp_path / "config.json").read_text() == saved
    picard_sweep.main(base + ["--J-up-to-L"])
    assert calls[-1][:2] == (list(range(1, 8)), list(range(1, 8)))
    with pytest.raises(SystemExit, match="different protocol"):
        picard_sweep.main(base + ["--J-up-to-L", "--seed", "123"])
    with pytest.raises(SystemExit, match="subset of the saved grid"):
        picard_sweep.main(base + ["--J-up-to-L", "--K-values", "8"])
    assert json.loads((tmp_path / "config.json").read_text())["seed"] == 20260714


def test_rules_can_join_an_existing_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        picard_sweep, "_run_epsilon",
        lambda args, out, cfg, eps, progress: calls.append(list(args.rules)),
    )
    base = ["--out", str(tmp_path), "--eps", "0.1", "--progress", "none", "--J-up-to-L"]
    config_path = tmp_path / "config.json"
    picard_sweep.main(base + ["--rules", "rmc", "d-grs"])
    paws = json.dumps({"paws": {"rank_policy": "max"}})
    picard_sweep.main(base + ["--rules", "paws", "--verifier-options", paws])
    assert calls[-1] == ["paws"]
    saved = json.loads(config_path.read_text())
    assert saved["rules"] == ["rmc", "d-grs", "paws"]
    assert saved["verifier_options"] == json.loads(paws)

    # Rerunning saved rules resumes them without changing the metadata.
    picard_sweep.main(base + ["--rules", "d-grs", "paws", "--verifier-options", paws])
    assert calls[-1] == ["d-grs", "paws"]
    assert json.loads(config_path.read_text()) == saved
    # A rule keeps the options of its first run, and the protocol still binds.
    with pytest.raises(SystemExit, match="verifier options"):
        picard_sweep.main(base + ["--rules", "paws"])
    with pytest.raises(SystemExit, match="different protocol"):
        picard_sweep.main(base + ["--rules", "resample", "--seed", "123"])
    with pytest.raises(SystemExit, match="distinct"):
        picard_sweep.main(base + ["--rules", "rmc", "rmc"])
    assert json.loads(config_path.read_text()) == saved


def test_runs_from_before_per_rule_directories_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(picard_sweep, "_run_epsilon", lambda *args: None)
    base = ["--out", str(tmp_path), "--eps", "0.1", "--progress", "none", "--J-up-to-L"]
    picard_sweep.main(base)
    config_path = tmp_path / "config.json"
    legacy = json.loads(config_path.read_text())
    legacy["schema_version"] = 3
    config_path.write_text(json.dumps(legacy))
    with pytest.raises(SystemExit, match="predates per-rule"):
        picard_sweep.main(base)


def test_a_rule_added_later_matches_one_run_with_every_rule(tmp_path):
    base = [
        "--eps", "0.1", "--dimension", "4", "--num-components", "2",
        "--num-steps", "7", "--mixture-seed", "11", "--K-values", "2",
        "--L-values", "2", "--J-values", "0", "2", "--replicates", "2",
        "--progress", "none",
    ]
    together, split = tmp_path / "together", tmp_path / "split"
    picard_sweep.main(
        base + ["--out", str(together), "--rules", "rmc", "d-grs", "--n-workers", "2"]
    )
    picard_sweep.main(base + ["--out", str(split), "--rules", "rmc"])
    rmc_cell = split / "eps0.1" / "K2_L2" / "rmc" / "J2"
    finished = (rmc_cell / "COMPLETE").stat().st_mtime_ns
    picard_sweep.main(base + ["--out", str(split), "--rules", "rmc", "d-grs"])
    assert (rmc_cell / "COMPLETE").stat().st_mtime_ns == finished

    for rule in ("rmc", "d-grs"):
        for J in ("J0", "J2"):
            cell = f"eps0.1/K2_L2/{rule}/{J}/samples.npz"
            with np.load(together / cell) as a, np.load(split / cell) as b:
                assert a.files == b.files
                for field in a.files:
                    np.testing.assert_array_equal(a[field], b[field])

    def trajectories(run):
        with (run / "eps0.1" / "trajectories.csv").open(newline="") as handle:
            return [
                {key: value for key, value in row.items() if key != "sampling_seconds"}
                for row in csv.DictReader(handle)
            ]

    assert trajectories(split) == trajectories(together)
    assert {row["rule"] for row in trajectories(split)} == {"rmc", "d-grs"}
    assert json.loads((split / "config.json").read_text())["rules"] == ["rmc", "d-grs"]


@pytest.mark.parametrize("update", ["drift", "increment"])
def test_picard_update_selects_the_refinement_callback(update):
    setting = models.build(dimension=6, num_components=3, num_steps=8, eps=0.06)
    sampler, _, refiner, _, _ = picard_sweep.build_sampler(
        setting, "d-grs", 2, 2, 1, {**_config(), "picard_update": update}
    )
    assert sampler.refinement_update_fn is refiner
    assert refiner.update_fn is picard_sweep.PICARD_UPDATES[update]
