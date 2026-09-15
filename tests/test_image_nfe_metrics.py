"""Saved counters support post-hoc NFE metrics independently of round counts."""
import numpy as np
import pytest

from specdiff import BatchedSpeculativeSampler, DelayedDriftProposal, DraftTree, ConstantSchedule, create_verifier
from test_deterministic import BoundedTarget
from experiments.images.run_common import metric_totals, add_metrics, summarise_metrics, isolated_dispersion
from experiments.images.plot_edm import summarise


def test_metric_persistence_includes_initialization_and_phase_counts():
    target = BoundedTarget(5)
    sampler = BatchedSpeculativeSampler(target, DelayedDriftProposal(target),
        ConstantSchedule(.2), DraftTree.uniform(2, 2), create_verifier("d-grs"), num_steps=5)
    result = sampler.sample(np.zeros((3, 2)), rng=np.random.default_rng(23))
    part = metric_totals(result, num_steps=5, deterministic_steps=0)
    assert part["target_calls_per_trajectory"] != part["rounds_per_trajectory"]
    assert part["proposal_target_calls"] == 1
    assert sum(part[k] for k in ("proposal_target_calls", "refinement_target_calls", "verification_target_calls")) == part["target_calls"]
    assert sum(part["accepted_per_trajectory"]) == part["accepted_levels"]
    assert part["verified_per_trajectory"] == [5, 5, 5]
    total = {}
    add_metrics(total, part)
    add_metrics(total, part)
    assert total["target_calls_per_trajectory"] == part["target_calls_per_trajectory"] * 2
    assert total["batch_sizes"] == [3, 3]
    assert total["target_calls_per_batch"] == [result.target_calls] * 2
    summary = summarise_metrics(total)
    assert summary["mean_isolated_speedup"] == pytest.approx(result.mean_isolated_speedup)
    assert summary["speedup"] == summary["end_to_end_speedup"] == result.speedup
    samples = [5 / c for c in total["target_calls_per_trajectory"]]
    assert summary["std_isolated_speedup"] == pytest.approx(np.std(samples, ddof=1))


def test_plot_prefers_direct_calls_but_supports_legacy_rounds():
    row = dict(dataset="coco2014", eps=.8, method="D-GRS", budget=6, L=2, K=2,
               speculative_steps=50, rounds=[20, 25], calls=[21, 26],
               run="test", cell="K2_L2", rule="d-grs")
    new = summarise([row], "mean_isolated_speedup", "budget", over="images")[0]
    old = summarise([{**row, "calls": []}], "mean_isolated_speedup", "budget", over="images")[0]
    assert new["mean"] == pytest.approx(np.mean([50/21, 50/26]))
    assert old["mean"] == pytest.approx(np.mean([50/20, 50/25]))


def test_mixing_incomplete_new_and_legacy_counts_is_refused():
    with pytest.raises(ValueError, match="incomplete"):
        isolated_dispersion(dict(target_calls_per_trajectory=[3],
            rounds_per_trajectory=[2, 2], sample_count=2, batches=1, baseline_calls=5))
