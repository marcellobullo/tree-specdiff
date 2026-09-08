"""The Gaussian-mixture churn kernel of ``experiments/gm/models.py``.

``TestFrozenDrift`` pins the velocity-freezing hooks; ``TestLazyMirror`` checks
that ``experiments/gm/lazy.py`` -- which re-implements the proposal and the
prefetch policies by hand -- still costs what the eager sampler costs, the
same z-test as ``experiments/gm/validate_lazy.py`` at fewer replicates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.gm import lazy, models
from specdiff import DelayedDriftProposal, DraftTree, SpeculativeSampler, create_verifier


class TestFrozenDrift:
    def test_freeze_recovers_the_velocity_and_apply_inverts_it(self):
        t = models.build(dimension=4, num_components=3, num_steps=20, eps=0.6).target
        x = np.random.default_rng(0).standard_normal((4, t.dimension))
        steps = (0, 3, 7, 12)
        m = t((0,) * 4, x, steps)
        v = t.freeze_drift(x, m, steps)
        for i, s in enumerate(steps):
            assert np.allclose(v[i], t.velocity(x[i : i + 1], t.sigmas[s + t.step_offset])[0])
        assert np.allclose(t.apply_drift(v, x, steps), m)

    def test_affine_reproduces_the_kernel(self):
        t = models.build(dimension=4, num_components=3, num_steps=20, eps=0.6).target
        x = np.random.default_rng(1).standard_normal((2, t.dimension))
        for step in (t.step_offset, t.step_offset + 9):
            a, b = t.affine(step)
            v = t.velocity(x, t.sigmas[step])
            assert np.allclose(a * x + b * v, t.kernel(x, step)[0])


class TestLazyMirror:
    SEED = 20260714

    @staticmethod
    def _seeds(K, L, t):
        return np.random.SeedSequence(TestLazyMirror.SEED, spawn_key=(K, L, t)).spawn(2)

    @pytest.mark.parametrize("rule,K,L", [("d-grs", 2, 3), ("rmc", 1, 3)])
    @pytest.mark.parametrize("prefetch", ["none", "parent", "nearest"])
    @pytest.mark.parametrize("leaves", [False, True])
    def test_lazy_costs_what_eager_costs(self, rule, K, L, prefetch, leaves):
        setting = models.build(dimension=4, num_components=3, num_steps=12, eps=0.25)
        sampler = SpeculativeSampler(
            target=setting.target, proposal=DelayedDriftProposal(setting.target),
            schedule=setting.schedule, tree=DraftTree.uniform(K, L),
            verifier=create_verifier(rule), num_steps=setting.num_steps,
            prefetch=prefetch, evaluate_leaves=leaves,
        )
        reps = 120
        eager, lazily = [], []
        for t in range(reps):
            i, r = self._seeds(K, L, t)
            eager.append(sampler.sample(
                setting.initial_state(np.random.default_rng(i)),
                rng=np.random.default_rng(r)).target_calls)
            i, r = self._seeds(K, L, t)
            lazily.append(lazy.simulate(
                setting, rule, K, L, setting.initial_state(np.random.default_rng(i)),
                np.random.default_rng(r), prefetch=prefetch, evaluate_leaves=leaves,
            ).target_calls)
        eager, lazily = np.array(eager, float), np.array(lazily, float)
        se = math.sqrt(eager.var(ddof=1) / reps + lazily.var(ddof=1) / reps)
        z = (lazily.mean() - eager.mean()) / se if se > 0 else 0.0
        assert abs(z) <= 4, f"eager {eager.mean():.2f} vs lazy {lazily.mean():.2f} (z={z:.2f})"
