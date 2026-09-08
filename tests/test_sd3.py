"""The SD3 latent-space adapter, on CPU and with no 16 GiB download.

The toy pipeline in `experiments/images/toy_sd3.py` makes each prompt name a
constant latent, so a finished trajectory lands *exactly* on that prompt's
value -- the last step is deterministic and pulls `x + dt (x - mu)/sigma` to
`mu` regardless of the path. This makes prompt routing directly testable with
an equality assertion.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

torch = pytest.importorskip("torch")

from specdiff import (  # noqa: E402
    BatchedSpeculativeSampler,
    DelayedDriftProposal,
    DraftTree,
    SpeculativeSampler,
    create_verifier,
)

from images import sd3_models as sd3  # noqa: E402
from images.toy_sd3 import ToySD3Pipeline, guided_mu, prompt_mu  # noqa: E402

PROMPTS = ["a cat", "a dog", "a boat"]
STEPS, EPS = 20, 0.25


def make(prompts=PROMPTS, *, guidance_scale=1.0, px=64, forward_batch=0, **kw):
    pipe = ToySD3Pipeline(resolution_px=px)
    den = sd3.SD3Denoiser(pipe, prompts, guidance_scale=guidance_scale,
                          resolution_px=px, **kw)
    return den, sd3.build(den, num_steps=STEPS, eps=EPS, forward_batch=forward_batch)


class TestSchedule:
    def test_matches_the_pixel_adapter_exactly(self):
        """Verify that the duplicated pixel and latent implementations remain equivalent.

        `sd3_models.py` copies `sigma_grid`/`churn_std_grid` from `models.py`
        rather than importing them, so the latent and pixel experiments stay
        independent. This is the guard that makes the duplication safe.
        """
        from images import models as pixel

        for shift in (1.0, 3.0):
            a = pixel.sigma_grid(STEPS, shift)
            b = sd3.sigma_grid(STEPS, shift)
            assert torch.equal(a, b)
            # Swept over s_noise as well as shift: comparing only the defaults
            # would let a one-sided edit to either copy's churn std through.
            for s_noise in (0.5, 1.0, 2.0):
                assert torch.equal(
                    pixel.churn_std_grid(a, EPS, s_noise=s_noise),
                    sd3.churn_std_grid(b, EPS, s_noise=s_noise)), (shift, s_noise)

    def test_non_positive_s_noise_is_refused(self):
        """The latent adapter refuses it the same way the pixel one does."""
        den, _ = make()
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="s_noise must be > 0"):
                sd3.build(den, num_steps=STEPS, eps=EPS, s_noise=bad)

    def test_sd3_defaults_to_shift_3(self):
        """SD3.5 ships shift=3.0 where EDM's flow-matching default is 1.0."""
        assert sd3.DEFAULT_SHIFT == 3.0
        assert not torch.equal(sd3.sigma_grid(STEPS), sd3.sigma_grid(STEPS, 1.0))

    def test_endpoints_are_still_the_two_deterministic_steps(self):
        _, s = make()
        assert s.deterministic_steps == (0, STEPS - 1)
        assert s.num_steps == STEPS - 2
        assert all(s.schedule(n) > 0.0 for n in range(s.num_steps))


class TestVelocity:
    def test_is_the_closed_form(self):
        """Also pins the timestep scaling: the adapter multiplies by 1000 on
        the way in, and the toy transformer divides by it. A slip either side
        shows up here as a wrong velocity, not as a crash."""
        den, _ = make()
        den.set_prompt_batch([0, 1, 2])
        x = torch.randn((3, *den.state_shape), generator=torch.Generator().manual_seed(0))
        t = torch.tensor([0.3, 0.5, 0.9])

        got = den.velocity(x, t, [0, 1, 2])
        mu = torch.tensor([prompt_mu(p) for p in PROMPTS]).view(-1, 1, 1, 1)
        want = (x - mu) / t.view(-1, 1, 1, 1)
        assert torch.allclose(got, want, atol=1e-5)

    def test_rows_of_one_call_carry_their_own_prompt(self):
        """Verify that one forward pass can use a distinct caption per row."""
        den, _ = make()
        den.set_prompt_batch([0, 1, 2])
        x = torch.zeros((3, *den.state_shape))
        t = torch.full((3,), 0.5)

        got = den.velocity(x, t, [0, 1, 2])
        for i in range(3):
            row = den.velocity(x[i : i + 1], t[i : i + 1], [i])
            assert torch.allclose(got[i : i + 1], row, atol=1e-6)
        assert not torch.allclose(got[0], got[1])

    def test_guidance_moves_the_target_predictably(self):
        den, _ = make(guidance_scale=7.0)
        den.set_prompt_batch([0])
        x = torch.zeros((1, *den.state_shape))
        t = torch.tensor([0.5])

        got = den.velocity(x, t, [0])
        mu = guided_mu(PROMPTS[0], 7.0)
        assert torch.allclose(got, (x - mu) / 0.5, atol=1e-4)

    def test_set_prompt_batch_selects_the_global_rows(self):
        """Two levels of indexing: global on upload, batch-local on the call."""
        den, _ = make()
        x = torch.zeros((2, *den.state_shape))
        t = torch.full((2,), 0.5)

        den.set_prompt_batch([2, 0])            # images 2 and 0, in that order
        got = den.velocity(x, t, [0, 1])        # batch-local
        mu = torch.tensor([prompt_mu(PROMPTS[2]), prompt_mu(PROMPTS[0])]).view(-1, 1, 1, 1)
        assert torch.allclose(got, (x - mu) / 0.5, atol=1e-5)

    def test_prompt_set_without_indices_is_refused(self):
        den, _ = make()
        den.set_prompt_batch([0, 1, 2])
        with pytest.raises(ValueError, match="indices_in_batch"):
            den.velocity(torch.zeros((1, *den.state_shape)), torch.tensor([0.5]))

    def test_a_single_prompt_needs_no_indices(self):
        den, _ = make(["a cat"])
        assert not den.per_sample_prompts
        v = den.velocity(torch.zeros((2, *den.state_shape)), torch.full((2,), 0.5))
        assert torch.allclose(v, (0.0 - prompt_mu("a cat")) / 0.5 * torch.ones_like(v),
                              atol=1e-5)


class TestTarget:
    def test_forward_batch_is_exact_and_free(self):
        _, whole = make()
        _, chunked = make(forward_batch=2)
        for s in (whole, chunked):
            s.target.denoiser.set_prompt_batch([0, 1, 2])
        x = torch.randn((3, *whole.state_shape), generator=torch.Generator().manual_seed(1))
        idx, steps = (0, 1, 2), (1, 4, 7)

        assert torch.allclose(whole.target(idx, x, steps),
                              chunked.target(idx, x, steps), atol=1e-5)
        # Chunking is a memory trade, not a cost one.
        assert whole.target.num_calls == chunked.target.num_calls == 1
        assert whole.target.num_states == chunked.target.num_states == 3

    def test_decode_applies_the_vae_scaling(self):
        den, _ = make(px=64)
        lat = torch.randn((2, *den.state_shape), generator=torch.Generator().manual_seed(2))
        px = den.decode_pixels(lat)

        vae = den.pipe.vae
        expected = (lat / vae.config.scaling_factor + vae.config.shift_factor)
        expected = expected.mean(dim=1, keepdim=True)
        expected = expected.repeat_interleave(8, dim=-1).repeat_interleave(8, dim=-2)
        assert torch.allclose(px, expected.repeat(1, 3, 1, 1), atol=1e-5)
        assert px.shape == (2, 3, 64, 64)


class TestSampling:
    @pytest.mark.parametrize("rule,tree", [
        ("rmc", DraftTree.chain(4)),
        ("d-grs", DraftTree.uniform(branching=2, lookahead=3)),
    ])
    def test_single_trajectory_lands_on_its_prompt(self, rule, tree):
        den, s = make(["a cat"])
        sampler = SpeculativeSampler(
            target=s.target, proposal=DelayedDriftProposal(s.target),
            schedule=s.schedule, tree=tree, verifier=create_verifier(rule),
            num_steps=s.num_steps, check_contract=True,
        )
        gen = torch.Generator().manual_seed(3)
        y, result = sd3.sample_trajectory(s, sampler, s.initial_state(gen),
                                          rng=gen, generator=gen)
        # The final step is deterministic and pulls x to mu exactly, whatever
        # path the speculation took.
        assert torch.allclose(y, torch.full_like(y, prompt_mu("a cat")), atol=1e-3)
        assert result.target_calls < s.num_steps

    def test_each_image_follows_its_own_prompt(self):
        """A batch of different captions through one batched sampler."""
        den, s = make()
        den.set_prompt_batch([0, 1, 2])
        sampler = BatchedSpeculativeSampler(
            target=s.target, proposal=DelayedDriftProposal(s.target),
            schedule=s.schedule, tree=DraftTree.uniform(branching=2, lookahead=3),
            verifier=create_verifier("d-grs"), num_steps=s.num_steps,
        )
        gen = torch.Generator().manual_seed(4)
        y0 = torch.randn((3, *s.state_shape), generator=gen)
        y, _ = sd3.sample_trajectory(s, sampler, y0, rng=gen, generator=gen)

        want = torch.tensor([prompt_mu(p) for p in PROMPTS]).view(-1, 1, 1, 1)
        assert torch.allclose(y, want.expand_as(y), atol=1e-3)

    def test_permuting_the_prompts_permutes_the_images(self):
        """Verify prompt routing by reversing prompt order with a fixed seed."""
        den, s = make()
        sampler = BatchedSpeculativeSampler(
            target=s.target, proposal=DelayedDriftProposal(s.target),
            schedule=s.schedule, tree=DraftTree.uniform(branching=2, lookahead=2),
            verifier=create_verifier("d-grs"), num_steps=s.num_steps,
        )

        def run(order):
            den.set_prompt_batch(order)
            gen = torch.Generator().manual_seed(5)
            y0 = torch.randn((3, *s.state_shape), generator=gen)
            y, _ = sd3.sample_trajectory(s, sampler, y0, rng=gen, generator=gen)
            return y.reshape(3, -1).mean(dim=1)

        forward = run([0, 1, 2])
        reversed_ = run([2, 1, 0])
        assert torch.allclose(forward, reversed_.flip(0), atol=1e-3)


class TestDriver:
    """`run_sd3.py`. Kept a separate file from `run_edm.py` by choice, so the
    pieces they duplicate are pinned against each other here."""

    @staticmethod
    def _args(tmp_path, **over):
        from images import run_sd3

        argv = ["--toy", "--toy-resolution", "64", "--no-accelerate",
                "--device", "cpu", "--rule", "d-grs", "--branching", "2",
                "--lookahead", "2", "--num-steps", str(STEPS), "--eps", str(EPS),
                "--guidance-scale", "1.0", "--out", str(tmp_path)]
        for k, v in over.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return run_sd3.parse_args(argv)

    def test_shard_and_match_logic_agree_with_the_pixel_driver(self):
        """Duplicated on purpose; this is what stops the copies drifting."""
        from images import run_edm, run_sd3

        for n, world in ((9, 2), (8, 4), (5, 5), (256, 4)):
            for rank in range(world):
                assert (run_sd3.shard_bounds(n, rank, world)
                        == run_edm.shard_bounds(n, rank, world))
        for K, L in ((2, 3), (4, 3), (3, 2)):
            tree = DraftTree.uniform(branching=K, lookahead=L)
            for match in ("verification", "budget"):
                assert (run_sd3.matched_chain_depth(tree, 98, match)
                        == run_edm.matched_chain_depth(tree, 98, match))

    def test_the_two_specs_share_their_common_parameters(self):
        """The other half of the duplication guard, for the parameter specs.

        Everything the two drivers have in common must mean the same thing in
        both -- same section, same type, same protocol-or-placement answer -- or
        one config file's `schedule` is another's `method`. Defaults are allowed
        to differ, but only the ones that differ for a reason: SD3 ships a
        shorter horizon, a shifted sigma grid, a bigger latent and a bigger
        default run.
        """
        from images import run_edm, run_sd3

        edm = {p.name: p for p in run_edm.PARAMS}
        sd3_ = {p.name: p for p in run_sd3.PARAMS}
        shared = sorted(set(edm) & set(sd3_))
        assert len(shared) > 15, "the specs have drifted apart entirely"

        for name in shared:
            a, b = edm[name], sd3_[name]
            assert (a.section, a.kind, a.where, a.choices) == \
                   (b.section, b.kind, b.where, b.choices), name

        assert {n for n in shared if edm[n].default != sd3_[n].default} == {
            "network",         # a .pkl path vs a hub id
            "num_steps",       # 100 vs 28
            "shift",           # 1.0 vs SD3.5's 3.0
            "toy_resolution",  # 16 vs 64
            "num_samples",     # 64 vs 256
            "device",          # "cpu" vs resolved in main()
        }

    def test_prompt_file_is_read_as_a_prefix(self, tmp_path):
        from images import run_sd3

        f = tmp_path / "p.txt"
        f.write_text("a cat\na dog\na boat\n")
        assert run_sd3.load_prompts(str(f), 2) == ["a cat", "a dog"]

        with pytest.raises(SystemExit, match="exceeds"):
            run_sd3.load_prompts(str(f), 4)

        blank = tmp_path / "blank.txt"
        blank.write_text("a cat\n\na dog\n")
        with pytest.raises(SystemExit, match="blank lines"):
            run_sd3.load_prompts(str(blank), 2)

    def test_starting_noise_does_not_depend_on_batching(self, tmp_path):
        """Image i's noise comes from (seed + i) alone.

        Otherwise the same seed at a different `--sample-batch` or process count
        starts from different states, and two arms are no longer paired.
        """
        from images import run_sd3

        f = tmp_path / "p.txt"
        f.write_text("\n".join(PROMPTS * 2) + "\n")
        args = self._args(tmp_path, prompts=str(f), num_samples=6)
        den = run_sd3.build_denoiser(args)
        s = run_sd3.build_setting(args, den)

        gen = torch.Generator().manual_seed(0)
        whole = run_sd3.initial_latents(args, s, den, 0, 6, gen)
        halves = torch.cat([
            run_sd3.initial_latents(args, s, den, first, 3, gen) for first in (0, 3)
        ])
        assert torch.equal(whole, halves)
        # ...and different images really do get different noise.
        assert not torch.allclose(whole[0], whole[1])

    def test_two_shards_merge_and_each_image_keeps_its_caption(self, tmp_path):
        from images import run_sd3
        from images.toy_sd3 import prompt_mu

        f = tmp_path / "p.txt"
        f.write_text("\n".join(PROMPTS * 2) + "\n")
        args = self._args(tmp_path, prompts=str(f), num_samples=6, sample_batch=3)
        den = run_sd3.build_denoiser(args)
        s = run_sd3.build_setting(args, den)
        tree = run_sd3.build_tree(args, s.num_steps)
        sampler = run_sd3.build_sampler(s, tree, args)

        for rank in range(2):
            start, count = run_sd3.shard_bounds(args.num_samples, rank, 2)
            run_sd3.generate_shard(args, s, sampler, den, start, count, tmp_path, rank)
        meta = run_sd3.merge_shards(args, s, tree, den, tmp_path)

        assert meta["num_samples"] == 6
        assert meta["num_processes"] == 2
        assert meta["image_shape"] == [3, 64, 64]
        assert meta["prompts_file"] == str(f)
        assert not list(tmp_path.glob("shard_*.pt"))

        # Each image must carry the caption its index names, across the shard
        # boundary -- rank 1 starts at image 3, and its prompt indices are
        # global while `indices_in_batch` is local.
        samples = torch.load(tmp_path / "samples.pt", weights_only=True)
        got = samples.float().div(127.5).sub(1.0).reshape(6, -1).mean(dim=1)
        vae = den.pipe.vae.config
        want = torch.tensor(
            [prompt_mu(p) / vae.scaling_factor + vae.shift_factor for p in PROMPTS * 2]
        ).clamp(-1.0, 1.0)
        assert torch.allclose(got, want, atol=1e-2)


class TestClipScoring:
    """`clip.py`'s bookkeeping, with a stand-in scorer.

    Downloading CLIP ViT-L/14 to check the encoder is not a good trade; what
    *can* be wrong here is which caption gets paired with which image, and the
    paired statistics -- both testable without any weights.
    """

    class _FakeScorer:
        """Scores an image by how close its mean is to the caption's value."""

        def scores(self, images, prompts, batch_size):
            mu = images.float().div(127.5).sub(1.0).reshape(images.shape[0], -1).mean(1)
            want = torch.tensor([prompt_mu(p) for p in prompts])
            return (100.0 - 100.0 * (mu - want).abs()).clamp(min=0.0)

    def _run_dir(self, tmp_path, prompts_file, num_samples):
        from images import run_sd3

        args = TestDriver._args(tmp_path, prompts=str(prompts_file),
                                num_samples=num_samples, sample_batch=num_samples)
        den = run_sd3.build_denoiser(args)
        s = run_sd3.build_setting(args, den)
        tree = run_sd3.build_tree(args, s.num_steps)
        sampler = run_sd3.build_sampler(s, tree, args)
        run_sd3.generate_shard(args, s, sampler, den, 0, num_samples, tmp_path, 0)
        run_sd3.merge_shards(args, s, tree, den, tmp_path)
        return tmp_path

    def test_captions_are_recovered_from_meta(self, tmp_path):
        from images import clip

        f = tmp_path / "p.txt"
        f.write_text("\n".join(PROMPTS) + "\n")
        run = self._run_dir(tmp_path, f, 3)
        assert clip.cell_prompts(run, 3, None) == PROMPTS

    def test_a_single_prompt_is_repeated(self, tmp_path):
        from images import clip

        (tmp_path / "meta.json").write_text(json.dumps({"prompt": "a cat"}))
        assert clip.cell_prompts(tmp_path, 3, None) == ["a cat"] * 3

    def test_neither_field_is_refused(self, tmp_path):
        from images import clip

        (tmp_path / "meta.json").write_text(json.dumps({"num_samples": 3}))
        with pytest.raises(SystemExit, match="neither"):
            clip.cell_prompts(tmp_path, 3, None)

    def test_blank_lines_are_refused(self, tmp_path):
        """One caption per line is only unambiguous if none can be empty."""
        from images import clip

        f = tmp_path / "p.txt"
        f.write_text("a cat\n\na dog\n")
        with pytest.raises(SystemExit, match="blank lines"):
            clip.cell_prompts(tmp_path, 2, str(f))

    def test_a_literal_caption_override_is_not_read_as_a_path(self, tmp_path):
        from images import clip

        assert clip.cell_prompts(tmp_path, 2, "a photo of a cat") == \
            ["a photo of a cat"] * 2

    def test_images_are_paired_with_their_own_caption(self, tmp_path):
        """Detect off-by-one errors in caption-to-image alignment."""
        from images import clip

        f = tmp_path / "p.txt"
        f.write_text("\n".join(PROMPTS) + "\n")
        run = self._run_dir(tmp_path, f, 3)
        samples = torch.load(run / "samples.pt", weights_only=True)

        scorer = self._FakeScorer()
        right = clip.summarise(run, scorer.scores(samples, PROMPTS, 2), "fake", None)
        rotated = [PROMPTS[1], PROMPTS[2], PROMPTS[0]]
        wrong = clip.summarise(run, scorer.scores(samples, rotated, 2), "fake", None)

        assert right["num_images"] == 3
        assert len(right["per_image"]) == 3
        assert right["clip_score_mean"] > wrong["clip_score_mean"] + 5.0

    def test_the_paired_delta_beats_the_unpaired_one(self, tmp_path):
        """Why per-image scores are kept at all.

        Two cells generated from the same (caption, seed) pairs differ by a
        small common shift plus large per-caption variation. The paired
        standard error sees through that variation; the unpaired one does not,
        which is how a real gap hides at realistic sample counts.
        """
        from images import clip

        gen = torch.Generator().manual_seed(0)
        base_scores = 20.0 + 5.0 * torch.randn(200, generator=gen)   # caption spread
        shifted = base_scores + 0.3                                   # a real gap

        a = clip.summarise(Path("a"), shifted, "fake", None)
        b = clip.summarise(Path("b"), base_scores, "fake", None)
        a["pairing_signature"] = b["pairing_signature"] = "same-pairs"
        d = clip.paired_delta(a, b)

        assert d["paired_mean_delta"] == pytest.approx(0.3, abs=1e-4)
        assert d["paired_sem"] < d["unpaired_sem"] / 100     # validates paired scoring
        assert abs(d["z"]) > 10                              # visible when paired

    def test_incomparable_cells_give_no_delta(self, tmp_path):
        from images import clip

        a = clip.summarise(Path("a"), torch.ones(5), "fake", None)
        b = clip.summarise(Path("b"), torch.ones(3), "fake", None)
        assert clip.paired_delta(a, b) is None


class TestCocoPrompts:
    """The caption set's two load-bearing properties: determinism and prefix
    stability. Both are what let a 500-sample sweep and a later 30k run be
    compared without re-deriving anything."""

    @staticmethod
    def _annotations(tmp_path, n=50):
        anns = []
        for img in range(n):
            # Several captions per image, deliberately out of id order, so the
            # "lowest annotation id" rule has something to pick.
            for k, aid in enumerate((300 + img, 100 + img, 200 + img)):
                anns.append({"image_id": img, "id": aid,
                             "caption": f"  caption {aid}\n for image {img} "})
        random.shuffle(anns)                      # file order must not matter
        tmp_path = Path(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        p = tmp_path / "captions.json"
        p.write_text(json.dumps({"annotations": anns}))
        return str(p)

    def test_num_is_a_true_prefix(self, tmp_path):
        from images.coco_prompts import select

        path = self._annotations(tmp_path)
        big, big_ids = select(path, 40, 0, seed=7)
        small, small_ids = select(path, 10, 0, seed=7)
        assert small == big[:10]
        assert small_ids == big_ids[:10]

    def test_file_order_does_not_change_the_set(self, tmp_path):
        """Sorting before the shuffle is what guarantees this."""
        from images.coco_prompts import select

        a, ids_a = select(self._annotations(tmp_path / "a"), 20, 0, seed=7)
        b, ids_b = select(self._annotations(tmp_path / "b"), 20, 0, seed=7)
        assert ids_a == ids_b and a == b

    def test_lowest_annotation_id_wins(self, tmp_path):
        from images.coco_prompts import select

        captions, ids = select(self._annotations(tmp_path), 5, 0, seed=7)
        for cap, img in zip(captions, ids):
            assert cap == f"caption {100 + img} for image {img}"

    def test_captions_are_whitespace_normalised(self, tmp_path):
        """One caption per line is only unambiguous if none contains a newline."""
        from images.coco_prompts import normalise, select

        captions, _ = select(self._annotations(tmp_path), 5, 0, seed=7)
        assert all("\n" not in c and c == c.strip() for c in captions)
        assert normalise("  a\n b  ") == "a b"

    def test_asking_for_more_than_exists_is_refused(self, tmp_path):
        from images.coco_prompts import select

        with pytest.raises(SystemExit, match="exceeds"):
            select(self._annotations(tmp_path, n=10), 20, 0, seed=7)


def test_paired_delta_refuses_different_pairing_signatures():
    from images import clip

    scores = torch.tensor([1.0, 2.0, 3.0])
    a = clip.summarise(Path("a"), scores, "fake", None)
    b = clip.summarise(Path("b"), scores, "fake", None)
    a["pairing_signature"], b["pairing_signature"] = "seed-1", "seed-2"
    assert clip.paired_delta(a, b) is None


def test_clip_cache_signature_tracks_model_prompts_and_samples(tmp_path):
    from images import clip

    cell = tmp_path / "cell"
    cell.mkdir()
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a cat\n")
    torch.save(torch.zeros((1, 3, 8, 8), dtype=torch.uint8), cell / "samples.pt")
    meta = {"prompts_file": str(prompts), "seed": 0,
            "prompt_seed_rule": "image i = line i"}
    (cell / "meta.json").write_text(json.dumps(meta))
    args = TestDriver._args(cell, prompts=str(prompts), num_samples=1)
    args.model, args.dtype = "model-a", "float32"

    first = clip.cache_signature(cell, args, meta)
    args.model = "model-b"
    assert clip.cache_signature(cell, args, meta) != first
    args.model = "model-a"
    prompts.write_text("a dog\n")
    assert clip.cache_signature(cell, args, meta) != first


class TestFrozenDrift:
    def test_freeze_recovers_the_velocity_and_apply_inverts_it(self):
        """The latent kernel freezes the velocity like the pixel one does."""
        den, s = make()
        t = s.target
        rows, steps = (0, 1, 2), (0, 5, 11)
        den.set_prompt_batch(list(rows))
        x = torch.randn((3, *s.state_shape), generator=torch.Generator().manual_seed(32))
        m = t(rows, x, steps)
        v = t.freeze_drift(x, m, steps)
        idx = torch.tensor([st + t.step_offset for st in steps])
        v_true = den.velocity(x, t.sigmas[idx].to(torch.float32), rows)
        assert torch.allclose(v, v_true, atol=1e-4, rtol=1e-5)
        assert torch.allclose(t.apply_drift(v, x, steps), m, atol=1e-6)
