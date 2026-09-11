"""PAWS wiring through every experiment family, without GPUs or checkpoints."""
import json

import numpy as np
import pytest

from experiments.gm import gm_sweep, lazy, models, picard_sweep
from experiments.verifier_config import configured_verifier, parse_verifier_options


OPTIONS = {"paws": {"rank_policy": "max", "residual_complement": "nearest_projection"}}


def test_per_rule_options_do_not_leak_to_other_verifiers():
    options = parse_verifier_options(json.dumps(OPTIONS))
    assert configured_verifier("paws", options).rank_policy == "max"
    assert configured_verifier("rmc", options).requires_chain
    for bad in ['[]', '{"unknown": {}}', '{"paws": {"temperature": 2}}']:
        with pytest.raises((ValueError, TypeError)):
            parse_verifier_options(bad)


@pytest.mark.parametrize("leaves", [False, True])
def test_gm_eager_and_lazy_accept_variant_options(leaves):
    setting = models.build(dimension=4, num_components=2, num_steps=8, eps=.2)
    sampler, tree, _ = gm_sweep.build_sampler(
        setting, "paws", 2, 3, "nearest", leaves, "verification", OPTIONS)
    assert tree.branching == 2 and tree.depth == 3
    assert sampler.verifier.residual_complement == "nearest_projection"
    initial = setting.initial_state(np.random.default_rng(12))
    eager = sampler.sample(initial, rng=np.random.default_rng(13))
    deferred = lazy.simulate(
        setting, "paws", 2, 3, initial, np.random.default_rng(13),
        evaluate_leaves=leaves, verifier_options=OPTIONS)
    assert np.isfinite(eager.sample).all()
    assert np.isfinite(deferred.sample).all()
    assert eager.target_calls >= 1 and deferred.target_calls >= 1


def test_picard_preserves_paws_tree_and_records_levels():
    cfg = dict(seed=123, eps=.2, match="verification", prefetch="nearest",
               evaluate_leaves=False, check_contract=True, verifier_options=OPTIONS)
    setting = models.build(dimension=4, num_components=2, num_steps=8, eps=.2)
    built = picard_sweep.build_sampler(setting, "paws", 2, 3, 1, cfg)
    bundle = picard_sweep.one_trajectory(setting, "paws", 2, 3, 1, 0, cfg, built)
    assert bundle["trajectory"]["chain_depth"] == ""
    assert bundle["trajectory"]["actual_proposal_budget"] == 14
    assert bundle["rounds"][-1]["cumulative_committed"] == setting.num_steps
    assert all(row["accepted"] for row in bundle["levels"] if row["level"] <= 1)


@pytest.mark.parametrize("driver", ["edm", "sd3"])
def test_image_cli_topology_options_and_signature(driver, tmp_path):
    pytest.importorskip("torch")
    from experiments.images import run_edm, run_sd3
    from experiments.images.run_common import IGNORED_IN_SIGNATURE

    runner = run_edm if driver == "edm" else run_sd3
    argv = ["--toy", "--out", str(tmp_path), "--rule", "paws",
            "--branching", "2", "--lookahead", "3",
            "--verifier-options", json.dumps(OPTIONS)]
    args = runner.parse_args(argv)
    assert "verifier_options" not in IGNORED_IN_SIGNATURE
    tree = runner.build_tree(args, 30)
    assert tree.branching == 2 and tree.depth == 3
    args.rule = "rmc"
    assert runner.build_tree(args, 30).depth == 7
    args.sampler["evaluate_leaves"] = True
    assert runner.build_tree(args, 30).depth == 14
    args.rule = "target"
    assert runner.build_tree(args, 30).depth == 1
    with pytest.raises(SystemExit):
        runner.parse_args(argv + ["--verifier-options", '{"paws":{"temperature":2}}'])


def test_plot_labels_include_paws():
    from experiments.gm import plot_gm
    from experiments.images import plot_edm
    assert "paws" in plot_gm.RULE_ORDER
    assert plot_edm.RULE_LABEL["paws"] == "PAWS"
