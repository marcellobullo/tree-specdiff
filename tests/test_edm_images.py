"""The EDM adapter of ``experiments/images/models.py``, on CPU and with no checkpoint.

Four things can be wrong in the port, and there is one test class for each:

``TestChangeOfVariables``
    ``D(x / (1 - t); t / (1 - t)) -> v = (x - D) / t``. Checked against a
    velocity computed directly on the interpolant, never through the denoiser,
    so agreement is evidence and not a tautology.
``TestSchedule``
    The transition std is state-independent (the rank-1 reduction's premise),
    matches the reference churn scheduler's, and vanishes at exactly the two
    endpoints ``build`` claims to strip.
``TestTarget``
    ``means`` is one batched call over rows at mixed steps, ``--forward-batch``
    chunking is exact, and neither inflates the NFE count.
``TestSampling``
    Both rules run end to end and reproduce the standard sampler's law.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pytest

def two_sample_ks(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, without scipy.

    `specdiff.testing._ks_statistic` rolls its own one-sample KS for the same
    reason: the library declares no scientific-stack dependency, and a test
    guarded by `importorskip` on one would skip silently rather than fail --
    which is exactly how this test came to pass locally and skip on a server.
    """
    a, b = sorted(float(x) for x in a), sorted(float(x) for x in b)
    n, m = len(a), len(b)
    i = j = 0
    stat = 0.0
    while i < n and j < m:
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        stat = max(stat, abs(i / n - j / m))
    return stat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

torch = pytest.importorskip("torch")

from specdiff import (  # noqa: E402
    DelayedDriftProposal,
    DraftTree,
    SpeculativeSampler,
    create_verifier,
    standard_sampler,
)

from images import models  # noqa: E402
from images.toy import GaussianEDMPrecond, analytic_velocity  # noqa: E402

EPS = 0.25
STEPS = 20


def make_denoiser(**kw) -> models.EDMDenoiser:
    return models.EDMDenoiser(GaussianEDMPrecond(**kw))


class TestChangeOfVariables:
    @pytest.mark.parametrize("dtype,rtol", [(torch.float64, 1e-12),
                                            (torch.float32, 1e-3)])
    def test_matches_interpolant_velocity(self, dtype, rtol):
        """Validate algebraic agreement at the precision supported by each dtype.

        ``(x - D) / t`` divides a cancellation by ``t``, so at ``t = 0.01`` a
        float32 denoiser's last bits become a ~1e-4 relative error in the
        velocity. That is a property of the parameterisation, not of this port
        -- the float64 arm is what shows the change of variables is right.
        """
        net = GaussianEDMPrecond(img_resolution=4).to(dtype)
        denoiser = models.EDMDenoiser(net, dtype=dtype)
        gen = torch.Generator().manual_seed(0)
        # Inside [t_min, t_max] = [1/1001, 80/81] so no clamping is in play;
        # the clamped ends are the subject of `test_t_is_clamped_at_both_ends`.
        t = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.98], dtype=dtype)
        x = torch.randn((5, 3, 4, 4), generator=gen).to(dtype)

        got = denoiser.velocity(x, t)
        want = analytic_velocity(net, x, t)
        assert got.dtype == dtype
        assert torch.allclose(got, want, rtol=rtol, atol=1e-6)

    def test_conditional_labels_select_a_class(self):
        net = GaussianEDMPrecond(img_resolution=4, label_dim=3)
        denoiser = models.EDMDenoiser(net)
        gen = torch.Generator().manual_seed(1)
        t = torch.full((4,), 0.4)
        x = torch.randn((4, 3, 4, 4), generator=gen)

        for label in range(3):
            labels = torch.full((4,), label, dtype=torch.long)
            got = denoiser.velocity(x, t, labels)
            want = analytic_velocity(net, x, t, labels)
            assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)
        # Different classes must actually differ, or the one-hot plumbing could
        # be dropping the label and the checks above would still pass.
        zeros = torch.zeros(4, dtype=torch.long)
        assert not torch.allclose(
            denoiser.velocity(x, t, zeros), denoiser.velocity(x, t, zeros + 1)
        )

    def test_rows_of_one_call_can_carry_different_classes(self):
        """One call, several images, several classes -- the batched case."""
        net = GaussianEDMPrecond(img_resolution=4, label_dim=3)
        denoiser = models.EDMDenoiser(net)
        x = torch.randn((3, 3, 4, 4), generator=torch.Generator().manual_seed(2))
        t = torch.full((3,), 0.4)
        mixed = torch.tensor([0, 1, 2])

        got = denoiser.velocity(x, t, mixed)
        for i, label in enumerate(mixed):
            row = denoiser.velocity(x[i : i + 1], t[i : i + 1], label.reshape(1))
            assert torch.allclose(got[i : i + 1], row, atol=1e-6)

    def test_t_is_clamped_at_both_ends(self):
        denoiser = models.EDMDenoiser(GaussianEDMPrecond(img_resolution=4))
        x = torch.randn((2, 3, 4, 4), generator=torch.Generator().manual_seed(2))
        # t = 1 is sigma = infinity, t = 0 is 0/0; both must return finite rows.
        v = denoiser.velocity(x, torch.tensor([0.0, 1.0]))
        assert torch.isfinite(v).all()
        # Clamping uses the *same* t for the network call and the division, so
        # an out-of-range t must give exactly the boundary's velocity. Reading
        # `1 - t` off an unclamped t here is the subtle way to get this wrong.
        t_max = 80.0 / 81.0
        assert torch.allclose(
            denoiser.velocity(x, torch.tensor([1.0, 1.0])),
            denoiser.velocity(x, torch.tensor([t_max, t_max])),
            atol=1e-6,
        )

    def test_label_on_unconditional_checkpoint_is_refused(self):
        # Silently ignoring the label would produce plausible images of the
        # wrong distribution, which no downstream metric would catch.
        with pytest.raises(ValueError, match="unconditional"):
            models.ChurnKernelTarget(
                make_denoiser(), models.sigma_grid(STEPS), EPS, class_labels=[3]
            )

    def test_conditional_checkpoint_without_labels_is_refused(self):
        """EDM's cond nets have no null class, so omitting labels is not
        "unconditional" -- it is a zero embedding and the wrong distribution."""
        target = models.ChurnKernelTarget(
            models.EDMDenoiser(GaussianEDMPrecond(img_resolution=4, label_dim=3)),
            models.sigma_grid(STEPS), EPS,
        )
        with pytest.raises(ValueError, match="no null class"):
            target((0,), torch.zeros((1, 3, 4, 4)), (1,))

    def test_out_of_range_label_is_refused(self):
        target = models.ChurnKernelTarget(
            models.EDMDenoiser(GaussianEDMPrecond(img_resolution=4, label_dim=3)),
            models.sigma_grid(STEPS), EPS,
        )
        with pytest.raises(ValueError, match=r"\[0, 3\)"):
            target.set_class_labels([0, 3])


class TestSchedule:
    def test_sigma_grid_matches_diffusers(self):
        diffusers = pytest.importorskip("diffusers")
        sched = diffusers.FlowMatchEulerDiscreteScheduler()
        sched.set_timesteps(STEPS)
        want = sched.sigmas.to(torch.float64)
        got = models.sigma_grid(STEPS, shift=float(sched.config.shift))
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-9)

    def test_std_vanishes_exactly_at_the_endpoints(self):
        std = models.churn_std_grid(models.sigma_grid(STEPS), EPS)
        assert float(std[0]) == 0.0          # sigma = 1, g diverges
        assert float(std[-1]) == 0.0         # sigma_next = 0
        assert (std[1:-1] > 0.0).all()

    def test_std_is_state_independent(self):
        # The premise of the rank-1 reduction (eqs. 8-11): the two kernels may
        # differ in mean but never in variance.
        target = models.ChurnKernelTarget(make_denoiser(), models.sigma_grid(STEPS), EPS)
        gen = torch.Generator().manual_seed(3)
        steps = (5, 5, 5)
        a = target.kernel(torch.randn((3, 3, 8, 8), generator=gen), steps)[1]
        b = target.kernel(100.0 * torch.randn((3, 3, 8, 8), generator=gen), steps)[1]
        assert torch.equal(a, b)

    def test_kernel_std_agrees_with_the_grid(self):
        sigmas = models.sigma_grid(STEPS)
        grid = models.churn_std_grid(sigmas, EPS)
        target = models.ChurnKernelTarget(make_denoiser(), sigmas, EPS)
        steps = tuple(range(STEPS))
        got = target.kernel(torch.zeros((STEPS, 3, 8, 8)), steps)[1]
        assert torch.allclose(got, grid.float(), atol=1e-7)

    def test_build_strips_and_reports_the_endpoints(self):
        s = models.build(make_denoiser(), num_steps=STEPS, eps=EPS)
        assert s.deterministic_steps == (0, STEPS - 1)
        assert s.num_steps == STEPS - 2
        assert s.total_steps == STEPS
        assert s.target.step_offset == 1
        # A TabulatedSchedule needs exactly N entries, and NoiseSchedule refuses
        # a non-positive scale, so every one of them must be live.
        assert all(s.schedule(n) > 0.0 for n in range(s.num_steps))
        with pytest.raises(IndexError):
            s.schedule(s.num_steps)

    def test_zero_churn_has_nothing_to_speculate(self):
        with pytest.raises(ValueError, match="nothing to speculate"):
            models.build(make_denoiser(), num_steps=STEPS, eps=0.0)


class TestTarget:
    def test_means_handles_rows_at_different_steps(self):
        sigmas = models.sigma_grid(STEPS)
        target = models.ChurnKernelTarget(make_denoiser(), sigmas, EPS)
        gen = torch.Generator().manual_seed(4)
        x = torch.randn((4, 3, 8, 8), generator=gen)
        steps = (1, 4, 4, 9)

        got = target((0,) * len(steps), x, steps)
        # One row at a time is the definition; the batched path must match it.
        want = torch.cat([target.kernel(x[i : i + 1], (s,))[0]
                          for i, s in enumerate(steps)])
        assert torch.allclose(got, want, atol=1e-6)

    def test_forward_batch_is_exact_and_free(self):
        sigmas = models.sigma_grid(STEPS)
        x = torch.randn((7, 3, 8, 8), generator=torch.Generator().manual_seed(5))
        steps = (1, 2, 3, 4, 5, 6, 7)

        whole = models.ChurnKernelTarget(make_denoiser(), sigmas, EPS)
        chunked = models.ChurnKernelTarget(make_denoiser(), sigmas, EPS, forward_batch=3)
        idx = (0,) * len(steps)
        assert torch.allclose(whole(idx, x, steps), chunked(idx, x, steps), atol=1e-6)
        # Chunking is a memory trade, not a cost one: it must not show up as NFE.
        assert whole.num_calls == chunked.num_calls == 1
        assert whole.num_states == chunked.num_states == 7

    def test_step_offset_shifts_into_the_full_grid(self):
        sigmas = models.sigma_grid(STEPS)
        x = torch.randn((1, 3, 8, 8), generator=torch.Generator().manual_seed(6))
        plain = models.ChurnKernelTarget(make_denoiser(), sigmas, EPS)
        shifted = models.ChurnKernelTarget(make_denoiser(), sigmas, EPS, step_offset=3)
        assert torch.allclose(shifted((0,), x, (2,)), plain((0,), x, (5,)), atol=1e-6)


class TestSampling:
    @pytest.mark.parametrize("rule,tree", [
        ("rmc", DraftTree.chain(4)),
        ("d-grs", DraftTree.uniform(branching=3, lookahead=2)),
    ])
    def test_runs_end_to_end_and_saves_calls(self, rule, tree):
        s = models.build(make_denoiser(img_resolution=8), num_steps=STEPS, eps=EPS)
        sampler = SpeculativeSampler(
            target=s.target, proposal=DelayedDriftProposal(s.target),
            schedule=s.schedule, tree=tree, verifier=create_verifier(rule),
            num_steps=s.num_steps, check_contract=True,
        )
        gen = torch.Generator().manual_seed(7)
        y, result = models.sample_trajectory(
            s, sampler, s.initial_state(gen), rng=gen, generator=gen
        )
        assert y.shape == s.state_shape
        assert torch.isfinite(y).all()
        assert result.target_calls < s.num_steps      # speculation bought something
        assert models.to_uint8(y).dtype == torch.uint8

    def test_matches_the_standard_sampler_in_law(self):
        """Verify that speculation preserves the target marginal distribution.

        Both arms sample the same 8x8x3 model over the same horizon; a
        two-sample KS test on a fixed linear projection of the final state is
        the cheap version of ``specdiff.testing.check_exactness`` applied to the
        assembled model rather than to a rule in isolation.
        """
        s = models.build(make_denoiser(img_resolution=4), num_steps=STEPS, eps=EPS)
        tree = DraftTree.uniform(branching=2, lookahead=3)

        def draw(sampler, seed, n):
            out = []
            for i in range(n):
                gen = torch.Generator().manual_seed(seed + i)
                y, _ = models.sample_trajectory(
                    s, sampler, s.initial_state(gen), rng=gen, generator=gen
                )
                out.append(float(y.flatten()[:16].sum()))
            return out

        spec = SpeculativeSampler(
            target=s.target, proposal=DelayedDriftProposal(s.target),
            schedule=s.schedule, tree=tree, verifier=create_verifier("d-grs"),
            num_steps=s.num_steps,
        )
        base = standard_sampler(s.target, s.schedule, num_steps=s.num_steps)

        # Disjoint seed blocks: shared seeds would couple the two samples and
        # make the KS test meaningless.
        n = 300
        stat = two_sample_ks(draw(spec, 1_000, n), draw(base, 500_000, n))
        # Two-sample critical value at alpha = 0.01: c(alpha) sqrt((n + m) / nm).
        crit = 1.63 * math.sqrt((n + n) / (n * n))
        assert stat <= crit, (
            f"speculative and standard laws differ (KS {stat:.4f} > {crit:.4f})"
        )


class TestBatched:
    """The same setting through ``BatchedSpeculativeSampler``.

    Worth its own class because the endpoints and the speculative window meet
    differently here: trajectories are in lockstep for the Euler prologue and
    epilogue but fall out of step inside the window, which is exactly the seam
    ``sample_trajectory`` has to get right.
    """

    def test_batched_run_matches_shapes_and_accounting(self):
        from specdiff import DelayedDriftProposal, BatchedSpeculativeSampler

        s = models.build(make_denoiser(img_resolution=8), num_steps=STEPS, eps=EPS)
        sampler = BatchedSpeculativeSampler(
            target=s.target, proposal=DelayedDriftProposal(s.target),
            schedule=s.schedule, tree=DraftTree.uniform(branching=2, lookahead=3),
            verifier=create_verifier("d-grs"), num_steps=s.num_steps,
        )
        gen = torch.Generator().manual_seed(8)
        y0 = torch.randn((5, *s.state_shape), generator=gen)

        out, result = models.sample_trajectory(s, sampler, y0, rng=gen, generator=gen)
        assert out.shape == (5, *s.state_shape)
        assert torch.isfinite(out).all()
        # Wall-clock speedup is bounded above by what the trajectories would
        # each manage alone; the gap is the straggler cost of sharing a batch.
        assert 1.0 < result.speedup <= result.mean_isolated_speedup + 1e-9
        assert 0.0 < result.occupancy <= 1.0

    def test_single_and_batched_endpoints_agree(self):
        """The Euler endpoints are deterministic, so batching cannot move them."""
        s = models.build(make_denoiser(img_resolution=4), num_steps=STEPS, eps=EPS)
        gen = torch.Generator().manual_seed(9)
        y = torch.randn((3, *s.state_shape), generator=gen)

        stacked = models.euler_steps(s.target, y, range(0, 1), gen)
        rowwise = torch.cat([
            models.euler_steps(s.target, y[i : i + 1], range(0, 1), gen) for i in range(3)
        ])
        assert torch.allclose(stacked, rowwise, atol=1e-6)


class TestConditional:
    """Class labels routed per image, end to end through the batched sampler.

    The toy denoiser's data distribution is ``N(mu_c, s^2 I)`` per class, with
    the class means far apart and the spread small, so a finished sample can be
    classified by nearest class mean. That turns "did image 3's label reach
    image 3's entries?" into an assertion -- which is the failure this whole
    contract change exists to prevent, and one that produces plausible images
    rather than an error when it goes wrong.
    """

    @staticmethod
    def _separable_setting(**kw):
        net = GaussianEDMPrecond(
            img_resolution=4, label_dim=3, data_std=0.2, mean_scale=1.0
        )
        return net, models.build(models.EDMDenoiser(net), num_steps=STEPS, eps=EPS, **kw)

    def test_each_image_is_sampled_from_its_own_class(self):
        from specdiff import BatchedSpeculativeSampler

        net, s = self._separable_setting()
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        s.target.set_class_labels(labels)

        sampler = BatchedSpeculativeSampler(
            target=s.target, proposal=DelayedDriftProposal(s.target),
            schedule=s.schedule, tree=DraftTree.uniform(branching=2, lookahead=3),
            verifier=create_verifier("d-grs"), num_steps=s.num_steps,
        )
        gen = torch.Generator().manual_seed(20)
        y0 = torch.randn((len(labels), *s.state_shape), generator=gen)
        out, _ = models.sample_trajectory(s, sampler, y0, rng=gen, generator=gen)

        means = net.class_means.reshape(3, -1)
        flat = out.reshape(len(labels), -1)
        nearest = torch.cdist(flat, means).argmin(dim=1)
        assert torch.equal(nearest, labels), f"got classes {nearest.tolist()}"

    def test_permuting_labels_permutes_the_images(self):
        """Verify label routing by reversing labels while holding the seed fixed."""
        from specdiff import BatchedSpeculativeSampler

        net, s = self._separable_setting()
        means = net.class_means.reshape(3, -1)

        def run(labels):
            s.target.set_class_labels(labels)
            sampler = BatchedSpeculativeSampler(
                target=s.target, proposal=DelayedDriftProposal(s.target),
                schedule=s.schedule, tree=DraftTree.uniform(branching=2, lookahead=2),
                verifier=create_verifier("d-grs"), num_steps=s.num_steps,
            )
            gen = torch.Generator().manual_seed(21)
            y0 = torch.randn((3, *s.state_shape), generator=gen)
            out, _ = models.sample_trajectory(s, sampler, y0, rng=gen, generator=gen)
            return torch.cdist(out.reshape(3, -1), means).argmin(dim=1)

        assert torch.equal(run(torch.tensor([0, 1, 2])), torch.tensor([0, 1, 2]))
        assert torch.equal(run(torch.tensor([2, 1, 0])), torch.tensor([2, 1, 0]))


class TestSharding:
    """Splitting a run across processes, without needing a second process.

    The distributed launch is `accelerate`'s job. What belongs to this repo is
    the arithmetic (who generates which images), the shard files, and the merge
    — and all three are testable in one process by calling the rank-parametric
    functions directly.
    """

    @staticmethod
    def _args(tmp_path, **over):
        from images import run_edm

        argv = ["--toy", "--toy-resolution", "8", "--toy-classes", "10",
                "--no-accelerate", "--rule", "d-grs", "--branching", "2",
                "--lookahead", "2", "--num-steps", str(STEPS),
                "--eps", str(EPS), "--out", str(tmp_path)]
        for k, v in over.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return run_edm.parse_args(argv)

    @pytest.mark.parametrize("num_samples,world", [(9, 2), (8, 4), (5, 5), (3, 4), (50, 1)])
    def test_blocks_are_contiguous_and_cover_everything(self, num_samples, world):
        from images.run_edm import shard_bounds

        blocks = [shard_bounds(num_samples, r, world) for r in range(world)]
        covered = [i for start, count in blocks for i in range(start, start + count)]
        assert covered == list(range(num_samples))          # contiguous, in rank order
        counts = [c for _, c in blocks]
        assert max(counts) - min(counts) <= 1               # balanced to within one

    def test_labels_do_not_depend_on_the_process_count(self):
        """Image i's class must come from (i, seed) alone.

        Otherwise the same seed at 1 GPU and at 4 GPUs conditions on different
        classes, and the two runs' FIDs are not comparable.
        """
        from images.run_edm import all_labels, shard_bounds

        denoiser = models.EDMDenoiser(GaussianEDMPrecond(img_resolution=8, label_dim=10))
        labels = all_labels("uniform", 12, denoiser, seed=3)
        assert torch.equal(labels, all_labels("uniform", 12, denoiser, seed=3))

        for world in (1, 2, 3, 4):
            rejoined = torch.cat([
                labels[s : s + c]
                for s, c in (shard_bounds(12, r, world) for r in range(world))
            ])
            assert torch.equal(rejoined, labels)

    def test_a_slow_rank_is_reported(self, tmp_path, capsys):
        """The whole batch waits on the slowest rank, so say so."""
        from images import run_edm

        args = self._args(tmp_path, num_samples=4)
        denoiser = run_edm.build_denoiser(args)
        setting = models.build(denoiser, num_steps=args.num_steps, eps=args.eps)
        tree = run_edm.build_tree(args, setting.num_steps)

        signature = run_edm.experiment_signature(args, setting, tree, denoiser)
        totals = {
            "baseline_calls": 1, "target_calls": 1,
            "end_to_end_baseline_calls": 1, "end_to_end_target_calls": 1,
            "isolated_speedup_sum": 2.0, "sample_count": 2,
            "occupancy_active": 2, "occupancy_slots": 2,
            "accepted_levels": 1, "verified_levels": 2,
            "target_states_evaluated": 1,
        }
        for rank, secs in ((0, 10.0), (1, 40.0)):        # rank 1 four times slower
            torch.save({"samples": torch.zeros((2, *setting.state_shape), dtype=torch.uint8),
                        "rank": rank, "start": rank * 2, "count": 2,
                        "run_signature": signature, "metric_totals": totals,
                        "seconds": secs},
                       tmp_path / f"shard_{rank:03d}.pt")

        meta = run_edm.merge_shards(args, setting, tree, denoiser, "none", tmp_path)
        assert meta["seconds"] == 40.0
        assert meta["seconds_per_rank"] == [10.0, 40.0]
        assert "4.00x the fastest" in capsys.readouterr().out

    def test_two_shards_merge_into_one_run(self, tmp_path):
        from images import run_edm

        args = self._args(tmp_path, num_samples=9, sample_batch=4)
        denoiser = run_edm.build_denoiser(args)
        setting = models.build(denoiser, num_steps=args.num_steps, eps=args.eps)
        tree = run_edm.build_tree(args, setting.num_steps)
        sampler = run_edm.build_sampler(setting, tree, args)
        mode = run_edm.resolve_label_mode(args, denoiser)
        labels = run_edm.all_labels(mode, args.num_samples, denoiser, args.seed)

        paths = []
        for rank in range(2):
            start, count = run_edm.shard_bounds(args.num_samples, rank, 2)
            paths.append(run_edm.generate_shard(
                args, setting, sampler, denoiser, labels, start, count,
                tmp_path, rank, silent_reporter(tmp_path, rank=rank, world=2,
                                                total=args.num_samples),
            ))
        assert [p.name for p in paths] == ["shard_000.pt", "shard_001.pt"]

        # Re-running must reuse, not regenerate: that is what makes a crashed
        # 50k run cost only its unfinished shard.
        before = paths[0].stat().st_mtime_ns
        run_edm.generate_shard(args, setting, sampler, denoiser, labels,
                               *run_edm.shard_bounds(args.num_samples, 0, 2),
                               tmp_path, 0, silent_reporter(tmp_path, world=2,
                                                            total=args.num_samples))
        assert paths[0].stat().st_mtime_ns == before

        meta = run_edm.merge_shards(args, setting, tree, denoiser, mode, tmp_path)
        assert meta["num_samples"] == 9
        assert meta["num_processes"] == 2
        # Per-rank timings survive the merge: wall clock is the slowest rank, so
        # a straggler has to stay diagnosable after the run.
        assert len(meta["seconds_per_rank"]) == 2
        assert meta["seconds"] == max(meta["seconds_per_rank"])
        assert not list(tmp_path.glob("progress_rank*"))
        samples = torch.load(tmp_path / "samples.pt", weights_only=True)
        assert samples.shape == (9, *setting.state_shape)
        assert samples.dtype == torch.uint8
        # Shards are consumed by the merge, so a resumed run does not re-merge
        # stale pieces alongside the finished file.
        assert not list(tmp_path.glob("shard_*.pt"))


class TestMatchedChain:
    """Sizing the RMC arm against a (K, L) tree.

    RMC is a single-proposal coupling, so it has no tree of its own; a sweep
    hands both rules the same (K, L) and the matching decides how long the
    chain is. Which protocol is used changes the answer by a factor of K, so it
    is worth pinning.
    """

    def test_verification_matching_equalises_the_target_batch(self):
        """Verify that `chain(m)` evaluates `m` verification nodes."""
        from images.run_edm import matched_chain_depth

        for K, L in ((2, 3), (3, 2), (2, 5), (4, 3)):
            tree = DraftTree.uniform(branching=K, lookahead=L)
            depth = matched_chain_depth(tree, 98, "verification")
            assert DraftTree.chain(depth).verification_budget() == tree.verification_budget()

    def test_budget_matching_equalises_the_proposal_budget(self):
        from images.run_edm import matched_chain_depth

        for K, L in ((2, 3), (3, 2), (4, 3)):
            tree = DraftTree.uniform(branching=K, lookahead=L)
            depth = matched_chain_depth(tree, 98, "budget")
            assert DraftTree.chain(depth).budget == tree.budget

    def test_the_two_protocols_differ_by_a_factor_of_k(self):
        """Verify that proposal- and verification-matched protocols differ by `K`."""
        from images.run_edm import matched_chain_depth

        tree = DraftTree.uniform(branching=4, lookahead=3)
        assert matched_chain_depth(tree, 98, "verification") == 21
        assert matched_chain_depth(tree, 98, "budget") == 84

    def test_both_clamp_to_the_horizon_and_converge(self):
        """A round truncates to min(depth, N - n), so depth beyond N is unreachable."""
        from images.run_edm import matched_chain_depth

        tree = DraftTree.uniform(branching=4, lookahead=4)      # B = 340, |I| = 85
        assert matched_chain_depth(tree, 20, "verification") == 20
        assert matched_chain_depth(tree, 20, "budget") == 20


class TestExperimentBookkeeping:
    def test_ratios_are_reconstructed_from_additive_counters(self):
        from images.run_common import add_metrics, summarise_metrics

        def counters(calls):
            return {
                "baseline_calls": 10, "target_calls": calls,
                "end_to_end_baseline_calls": 12,
                "end_to_end_target_calls": calls + 2,
                "isolated_speedup_sum": 2.0, "sample_count": 1,
                "occupancy_active": 1, "occupancy_slots": 2,
                "accepted_levels": 1, "verified_levels": 2,
                "target_states_evaluated": 3,
            }

        total = {}
        add_metrics(total, counters(1))
        add_metrics(total, counters(9))
        summary = summarise_metrics(total)
        assert summary["speedup"] == pytest.approx(2.0)  # 20 baseline / 10 actual
        assert summary["speedup"] != pytest.approx((10.0 + 10.0 / 9.0) / 2)
        assert summary["acceptance_rate"] == 0.5

    def test_changed_configuration_refuses_a_reused_shard(self, tmp_path):
        from images import run_edm

        args = TestSharding._args(tmp_path, num_samples=2, sample_batch=2)
        denoiser = run_edm.build_denoiser(args)
        setting = models.build(denoiser, num_steps=args.num_steps, eps=args.eps)
        tree = run_edm.build_tree(args, setting.num_steps)
        sampler = run_edm.build_sampler(setting, tree, args)
        mode = run_edm.resolve_label_mode(args, denoiser)
        labels = run_edm.all_labels(mode, args.num_samples, denoiser, args.seed)
        run_edm.generate_shard(args, setting, sampler, denoiser, labels,
                               0, 2, tmp_path, 0, silent_reporter(tmp_path))

        args.seed += 1
        with pytest.raises(SystemExit, match="incompatible"):
            run_edm.generate_shard(args, setting, sampler, denoiser, labels,
                                   0, 2, tmp_path, 0, silent_reporter(tmp_path))

    def test_checkout_identity_ignores_git_and_pycache_churn(self, tmp_path):
        """A `git fetch` in the EDM checkout must not expire a resumable shard.

        The identity is there to catch a change to the *source* between the run
        that wrote a shard and the run that reuses it. .git/ and __pycache__
        move on their own: one fetch mid-sweep rewrote .git/FETCH_HEAD and
        every shard on disk was refused, with the code the run executes
        untouched.
        """
        from images.run_common import file_identity

        repo = tmp_path / "edm"
        (repo / ".git").mkdir(parents=True)
        (repo / "torch_utils" / "__pycache__").mkdir(parents=True)
        (repo / "generate.py").write_text("x = 1\n")
        (repo / ".git" / "FETCH_HEAD").write_text("abc\tnot-for-merge\n")
        (repo / "torch_utils" / "__pycache__" / "misc.cpython-310.pyc").write_bytes(b"\0")

        before = file_identity(str(repo))
        assert before["files"] == 1                   # the source file, and nothing else

        (repo / ".git" / "FETCH_HEAD").write_text("def\tnot-for-merge\tlonger\n")
        (repo / "torch_utils" / "__pycache__" / "misc.cpython-310.pyc").write_bytes(b"\0\1")
        assert file_identity(str(repo)) == before

        # Sizes differ, so this holds whatever the filesystem's mtime resolution.
        (repo / "generate.py").write_text("x = 222\n")
        assert file_identity(str(repo)) != before

    def test_zero_work_shard_is_refused(self, tmp_path):
        from images import run_edm

        args = TestSharding._args(tmp_path, num_samples=1)
        denoiser = run_edm.build_denoiser(args)
        setting = models.build(denoiser, num_steps=args.num_steps, eps=args.eps)
        tree = run_edm.build_tree(args, setting.num_steps)
        sampler = run_edm.build_sampler(setting, tree, args)
        with pytest.raises(SystemExit, match="at least one"):
            run_edm.generate_shard(args, setting, sampler, denoiser, None,
                                   1, 0, tmp_path, 1, silent_reporter(tmp_path))

    def test_grid_keeps_a_non_square_tail(self, tmp_path):
        import PIL.Image
        from images.run_common import save_grid

        path = tmp_path / "grid.png"
        save_grid(torch.zeros((2, 3, 8, 8), dtype=torch.uint8), path)
        assert PIL.Image.open(path).size == (16, 8)

    def test_ffhq_cache_identity_includes_the_data_source(self, tmp_path):
        from argparse import Namespace
        from images.fid import real_cache_signature

        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(); b.mkdir()
        (a / "one.png").write_bytes(b"a")
        (b / "one.png").write_bytes(b"b")
        args = Namespace(dataset="ffhq", data=str(a), num_real=1)
        first = real_cache_signature(args, 64)
        args.data = str(b)
        assert real_cache_signature(args, 64) != first


def silent_reporter(out, **over):
    """A reporter that writes its file but draws nothing, for driver tests."""
    from images.run_common import ProgressReporter

    kwargs = dict(rank=0, world=1, total=2, mode="none")
    kwargs.update(over)
    return ProgressReporter(out, **kwargs)


class TestProgress:
    """The progress line is bookkeeping: it must aggregate, and never raise."""

    def _reporter(self, out, **over):
        from images.run_common import ProgressReporter

        kwargs = dict(rank=0, world=2, total=10, label="d-grs", mode="plain",
                      write_every_s=0.0, plain_every_s=0.0)
        kwargs.update(over)
        return ProgressReporter(out, **kwargs)

    def test_rank_zero_counts_its_peers(self, tmp_path):
        """The bar is the job's, not one process's: peers' files are summed in."""
        peer = self._reporter(tmp_path, rank=1, mode="none")
        peer.update(4, 5, force=True)

        reporter = self._reporter(tmp_path)
        reporter.update(3, 5, force=True)
        done, rate = reporter._global()
        assert done == pytest.approx(7.0)                  # 3 of its own + 4 of rank 1
        assert rate > 0.0

    def test_a_finished_peer_adds_no_throughput(self, tmp_path):
        """Otherwise the ETA assumes an idle rank is still producing images."""
        def peer(finished):
            (tmp_path / "progress_rank001.json").write_text(json.dumps(
                {"rank": 1, "done": 4, "of": 5, "img_per_s": 7.0,
                 "finished": finished}
            ))

        reporter = self._reporter(tmp_path)
        reporter.started = time.time() - 10.0        # a stable own rate to compare
        reporter.update(1, 5, force=True)

        peer(False)
        _, working = reporter._global()
        peer(True)
        _, idle = reporter._global()
        assert working - idle == pytest.approx(7.0, rel=1e-3)

    def test_a_partly_written_peer_file_is_skipped(self, tmp_path):
        """Peers write while rank 0 reads; a torn frame must not end the run."""
        (tmp_path / "progress_rank001.json").write_text('{"done": 4, "of"')

        reporter = self._reporter(tmp_path)
        reporter.update(2, 5, force=True)
        done, _ = reporter._global()
        assert done == pytest.approx(2.0)

    def test_an_unwritable_directory_does_not_raise(self, tmp_path):
        """Hours of samples must not be lost to a full or read-only filesystem."""
        reporter = self._reporter(tmp_path / "missing", mode="none")
        reporter.update(1, 5, force=True)                  # no exception

    def test_in_flight_work_moves_the_line(self, tmp_path, capsys):
        """A run of a few large batches would otherwise sit still for minutes."""
        reporter = self._reporter(tmp_path, world=1, stream=sys.stdout)
        reporter.update(0, 10, in_flight=2.5, force=True)
        assert "2/10 img" in capsys.readouterr().out

    def test_progress_choice_does_not_invalidate_a_shard(self, tmp_path):
        """--progress is display; a shard made under one is reusable under any."""
        from images import run_edm

        args = TestSharding._args(tmp_path, num_samples=2)
        denoiser = run_edm.build_denoiser(args)
        setting = models.build(denoiser, num_steps=args.num_steps, eps=args.eps)
        tree = run_edm.build_tree(args, setting.num_steps)

        signature = run_edm.experiment_signature(args, setting, tree, denoiser)
        args.progress = "none"
        assert run_edm.experiment_signature(args, setting, tree, denoiser) == signature

    def test_the_sampler_reports_every_round_monotonically(self):
        """The hook's ratio must rise to exactly 1: it is what fills the bar."""
        from specdiff import BatchedSpeculativeSampler, DelayedDriftProposal

        denoiser = models.EDMDenoiser(GaussianEDMPrecond(img_resolution=8))
        setting = models.build(denoiser, num_steps=STEPS, eps=EPS)
        sampler = BatchedSpeculativeSampler(
            target=setting.target, proposal=DelayedDriftProposal(setting.target),
            schedule=setting.schedule,
            tree=DraftTree.uniform(branching=2, lookahead=2),
            verifier=create_verifier("d-grs"), num_steps=setting.num_steps,
        )
        seen = []
        generator = torch.Generator().manual_seed(0)
        y0 = torch.randn((3, *setting.state_shape), generator=generator)
        models.sample_trajectory(
            setting, sampler, y0, rng=generator, generator=generator,
            on_round=lambda taken, total: seen.append((taken, total)),
        )

        assert seen, "no round was reported"
        assert [t for t, _ in seen] == sorted(t for t, _ in seen)
        assert {total for _, total in seen} == {3 * setting.num_steps}
        assert seen[-1][0] == seen[-1][1]                   # ends exactly full


def test_merge_refuses_unexpected_extra_rank(tmp_path):
    from images.run_common import load_shards

    for rank in range(3):
        (tmp_path / f"shard_{rank:03d}.pt").touch()
    with pytest.raises(SystemExit, match="expected shards"):
        load_shards(tmp_path, signature={}, num_samples=2, world=2)
