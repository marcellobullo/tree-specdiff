"""Stable Diffusion 3.5 behind specdiff's ``TargetTransition``, in latent space.

The pixel-space adapter is ``models.py``; this is its latent-space counterpart,
kept separate because their operational requirements differ even though the
churn equations are identical:

* states are VAE **latents** ``(16, px/8, px/8)``, not images, so a decode step
  stands between the sampler and anything that looks at pixels;
* conditioning is **text**, one caption per image, encoded once up front;
* classifier-free guidance **doubles every forward**, so the memory a round
  needs is twice what the node count suggests.

Unlike EDM there is no change of variables: SD3 is trained on the same linear
flow-matching interpolant specdiff samples, so the transformer's output *is* the
velocity. This module applies classifier-free guidance and the churn transition.

Prompts
-------
``prompt`` may be a list, one caption per image. Every caption is encoded once
into a **CPU-resident** table and selected per latent by index -- at seq=333
a single embedding is ~2.7 MB, so a 30k COCO set is 82 GB and cannot live on
the GPU. :meth:`SD3Denoiser.set_prompt_batch` uploads only the rows a batch
needs; ``indices_in_batch`` then selects within them.

The second indexing level uses the batch-local ``indices_in_batch`` already
provided by specdiff, so prompt routing requires no sampler changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch

from specdiff import NoiseSchedule, TabulatedSchedule, TargetTransition

__all__ = [
    "NUM_TRAIN_TIMESTEPS",
    "sigma_grid",
    "churn_std_grid",
    "SD3Denoiser",
    "SD3ChurnTarget",
    "Setting",
    "build",
    "euler_steps",
    "sample_trajectory",
    "to_uint8",
]

NUM_TRAIN_TIMESTEPS = 1000
_SIGMA_GUARD = 1e-6

# SD3.5's scheduler ships shift=3.0, where EDM's flow-matching default is 1.0.
DEFAULT_SHIFT = 3.0


def sigma_grid(num_steps: int, shift: float = DEFAULT_SHIFT) -> torch.Tensor:
    """The ``(num_steps + 1,)`` sigma grid, terminal 0 included.

    Identical to ``models.sigma_grid`` apart from the default shift; see this
    module's docstring on why it is copied rather than imported.
    """
    timesteps = torch.linspace(NUM_TRAIN_TIMESTEPS, 1.0, num_steps, dtype=torch.float64)
    sigmas = timesteps / NUM_TRAIN_TIMESTEPS
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)])


def churn_std_grid(
    sigmas: torch.Tensor, eps: float, *, s_noise: float = 1.0
) -> torch.Tensor:
    """Per-step transition std ``(num_steps,)``; 0 marks a deterministic step."""
    sigma, sigma_next = sigmas[:-1], sigmas[1:]
    dt = sigma_next - sigma  # < 0
    g2 = 2.0 * sigma / (1.0 - sigma).clamp_min(_SIGMA_GUARD)
    std = eps * torch.sqrt(g2.clamp_min(0.0)) * torch.sqrt(-dt) * s_noise
    active = (
        (eps > 0.0)
        & (sigma > _SIGMA_GUARD)
        & (sigma < 1.0 - _SIGMA_GUARD)
        & (sigma_next > 0.0)
    )
    return torch.where(active, std, torch.zeros_like(std))


def prompt_cache_key(
    *, model_id, prompts, negative_prompt, do_cfg, dtype, encode_device,
    encode_batch,
) -> str:
    """Identity of one encoded caption table.

    Everything that can change the numbers is in the key, so a hit is the same
    computation and never merely the same captions. ``encode_device`` and
    ``encode_batch`` are in it for the reason ``--forward-batch`` is held fixed
    across a comparison set: the text encoders are not bit-reproducible across
    devices or batch shapes.
    """
    captions = hashlib.sha256("\n".join(prompts).encode()).hexdigest()
    identity = json.dumps(
        {"v": 1, "model": str(model_id), "captions": captions,
         "count": len(prompts), "negative": negative_prompt,
         "cfg": bool(do_cfg), "dtype": str(dtype),
         "encode_device": str(encode_device), "encode_batch": int(encode_batch)},
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def read_prompt_cache(directory, key: str):
    """The cached tables, or ``None`` on any miss.

    Absent, stale and unreadable all read as a miss, because the answer to each
    is the same -- encode it again. A half-written file from a killed run must
    not be an error a whole sweep dies on.
    """
    path = Path(directory) / f"sd3-prompts-{key}.pt"
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                                          # noqa: BLE001
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    return (payload["pos"], payload["pool"], payload["neg"], payload["neg_pool"])


def write_prompt_cache(directory, key: str, tables) -> None:
    """Write the tables atomically, so a reader never sees a partial file.

    Ranks of one job encode the same captions and race to write them. The
    rename is what makes that harmless: the loser overwrites identical bytes.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sd3-prompts-{key}.pt"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    pos, pool, neg, neg_pool = tables
    torch.save({"key": key, "pos": pos, "pool": pool,
                "neg": neg, "neg_pool": neg_pool}, tmp)
    os.replace(tmp, path)


class SD3Denoiser:
    """An SD3.5 pipeline exposed as a guided velocity field over latents.

    Mirrors ``StableDiffusion3Pipeline.__call__``'s denoising step -- the
    CFG stack, ``chunk(2)``, and the guidance combination -- with one departure
    that speculative verification requires: a **per-latent timestep vector**
    instead of the pipeline's scalar ``t.expand(2B)``, so parents sitting at
    different steps go through one forward.
    """

    def __init__(
        self,
        pipe,
        prompt: str | Sequence[str],
        *,
        negative_prompt: str = "",
        guidance_scale: float = 7.0,
        resolution_px: int = 512,
        encode_batch: int = 16,
        free_text_encoders: bool = True,
        device: "str | torch.device | None" = None,
        encode_device: "str | torch.device | None" = None,
        prompt_cache: "str | Path | None" = None,
        cache_id: str = "",
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        # `device` is where sampling happens. It must be given whenever the
        # transformer has not been moved there yet -- the low-memory path below
        # moves it only after the text encoders are gone, and `pipe.device`
        # would report the CPU until then.
        self.device = torch.device(device) if device is not None else pipe.device
        # Off the transformer, not the pipeline: `pipe.dtype` is the *first*
        # module's, so a float32 text encoder would redefine what latents are
        # cast to. The toy pipeline has no transformer dtype, so it falls back.
        self.dtype = getattr(pipe.transformer, "dtype", pipe.dtype)
        # Where the text encoders run, which need not be where sampling does.
        self.encode_device = (torch.device(encode_device)
                              if encode_device is not None else self.device)
        self.guidance_scale = float(guidance_scale)
        self.do_cfg = self.guidance_scale > 1.0
        self.resolution_px = int(resolution_px)
        # Conditioning is textual. Kept at 0 so anything written against the
        # pixel adapter's class-conditional checks stays uniform.
        self.num_classes = 0

        latent_size = resolution_px // pipe.vae_scale_factor
        patch = pipe.transformer.config.patch_size
        if latent_size % patch != 0:
            raise ValueError(
                f"resolution_px={resolution_px} gives latent size {latent_size}, "
                f"not a multiple of the transformer patch size {patch}"
            )
        self.state_shape: Tuple[int, int, int] = (
            int(pipe.transformer.config.in_channels), latent_size, latent_size
        )

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            raise ValueError("at least one prompt is required")
        self.prompts = prompts
        self.per_sample_prompts = len(prompts) > 1

        key = prompt_cache_key(
            model_id=cache_id, prompts=prompts, negative_prompt=negative_prompt,
            do_cfg=self.do_cfg, dtype=self.dtype,
            encode_device=self.encode_device.type, encode_batch=encode_batch,
        )
        tables = read_prompt_cache(prompt_cache, key) if prompt_cache else None
        self.prompt_cache_hit = tables is not None
        if tables is None:
            tables = self._encode(pipe, prompts, negative_prompt, encode_batch)
            if prompt_cache:
                write_prompt_cache(prompt_cache, key, tables)

        # (P, seq, dim) and (P, pooled_dim), both on the CPU: see the module
        # docstring on why the caption table never goes to the GPU whole.
        self._pos_table, self._pool_table, neg, neg_pool = tables
        # The negative is a single row, expanded per call, so it does go there.
        # It is already in `dtype`, which a float32 CPU encode makes necessary.
        self.neg = None if neg is None else neg.to(self.device)
        self.neg_pool = None if neg_pool is None else neg_pool.to(self.device)
        if self.per_sample_prompts:
            self.pos = self.pos_pool = None   # uploaded per batch
        else:
            self.pos = self._pos_table.to(self.device)
            self.pos_pool = self._pool_table.to(self.device)

        if free_text_encoders:
            self.free_text_encoders()

        # The sampling modules move last, so the peak on `device` never holds a
        # text encoder that has already been dropped. A no-op when the caller
        # placed the whole pipeline itself.
        for name in ("transformer", "vae"):
            module = getattr(pipe, name, None)
            if isinstance(module, torch.nn.Module) and module.device != self.device:
                module.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        encode_device: Optional[str] = None,
        **kwargs,
    ) -> "SD3Denoiser":
        """Build from a hub id or a local diffusers directory.

        ``encode_device`` is where the text encoders run. ``None`` puts the
        whole pipeline on ``device``, which needs all 16.3 GiB at once.

        Naming another device -- in practice ``"cpu"`` -- loads the text
        encoders there and leaves only the transformer and the VAE on
        ``device``. The peak drops from ~16.3 GiB to ~4.8 GiB, which is what
        makes a 10 GiB card usable. Nothing about the sampling changes: the
        text encoders run once, before the first latent exists, and are freed
        immediately afterwards.

        A CPU text encoder is loaded in float32. bfloat16 has no CPU kernels
        worth the name -- T5-XXL measures ~3x slower in it -- and the
        embeddings are cast back to ``dtype`` before the transformer sees them.
        """
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        if encode_device is None:
            pipe.to(device)
            return cls(pipe, device=device, cache_id=model_id, **kwargs)

        where = torch.device(encode_device)
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(pipe, name, None)
            if encoder is not None:
                encoder.to(where, torch.float32 if where.type == "cpu" else dtype)
        # The transformer and the VAE stay put; __init__ moves them once the
        # text encoders are gone.
        return cls(pipe, device=device, encode_device=where,
                   cache_id=model_id, **kwargs)

    def free_text_encoders(self) -> None:
        """Drop the text encoders once every prompt is encoded.

        11.2 of the 16.3 GiB an SD3.5-medium pipeline holds is T5-XXL plus the
        two CLIPs; the transformer that does the sampling is 4.5 GiB. After
        pre-encoding they are never called again, so this is close to a 3x cut
        in resident memory -- the headroom a deep verification tree needs.
        Irreversible for this instance: ``encode_prompt`` fails afterwards.
        """
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            if getattr(self.pipe, name, None) is not None:
                setattr(self.pipe, name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode(self, pipe, prompts, negative_prompt, encode_batch):
        """Encode every caption plus one negative; CPU tensors in ``self.dtype``.

        The negative is encoded **once**, on its own. Asking ``encode_prompt``
        for it alongside the captions makes diffusers repeat it for every row of
        every batch -- 100 captions cost 200 sequences -- and every row but the
        first is then discarded. Encoding it separately is the same function on
        the same text, and halves the work.

        It is the same computation, not the same bits: a batch of one sums in a
        different order from a batch of sixteen. Measured on SD3.5-medium the
        gap is ~6e-5 against embeddings of magnitude ~850, which is 46 of
        1,363,968 elements landing one ulp apart once cast to bfloat16. Rows of
        a single batch already differ by ~3e-6 among themselves for the same
        reason. This is the sense in which ``--forward-batch`` is exact too.
        """
        pos, pool = [], []
        with torch.no_grad():
            for i in range(0, len(prompts), encode_batch):
                p, _, pp, _ = pipe.encode_prompt(
                    prompt=prompts[i : i + encode_batch],
                    prompt_2=None,
                    prompt_3=None,
                    do_classifier_free_guidance=False,
                    device=self.encode_device,
                    num_images_per_prompt=1,
                )
                pos.append(p.to("cpu", self.dtype))
                pool.append(pp.to("cpu", self.dtype))
            neg = neg_pool = None
            if self.do_cfg:
                n, _, np_, _ = pipe.encode_prompt(
                    prompt=[negative_prompt],
                    prompt_2=None,
                    prompt_3=None,
                    do_classifier_free_guidance=False,
                    device=self.encode_device,
                    num_images_per_prompt=1,
                )
                neg = n.to("cpu", self.dtype)
                neg_pool = np_.to("cpu", self.dtype)
        return torch.cat(pos), torch.cat(pool), neg, neg_pool

    def set_prompt_batch(self, global_indices: Sequence[int]) -> None:
        """Upload this batch's prompt rows, in image order.

        Call once per sampling batch with the **global** prompt indices of its
        images. :meth:`velocity` is then given ``indices_in_batch`` -- indices
        into what was just uploaded, which is exactly what specdiff already
        passes -- so no further translation is needed.
        """
        if not self.per_sample_prompts:
            return
        idx = torch.as_tensor(list(global_indices), dtype=torch.long)
        self.pos = self._pos_table[idx].to(self.device)
        self.pos_pool = self._pool_table[idx].to(self.device)

    @property
    def scheduler_config(self) -> dict:
        return dict(self.pipe.scheduler.config)

    @torch.no_grad()
    def velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        indices_in_batch: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Guided velocity at ``x``, one row per element of ``t``.

        ``t`` is the interpolant coefficient (what the schedule calls sigma);
        the transformer wants it on the training timestep scale, so it is
        multiplied by ``num_train_timesteps`` here rather than at the call site.
        """
        rows = x.shape[0]
        if self.per_sample_prompts:
            if indices_in_batch is None:
                raise ValueError(
                    "this model holds a prompt set; pass indices_in_batch (which "
                    "image each entry belongs to)"
                )
            if self.pos is None:
                raise RuntimeError("call set_prompt_batch() before velocity()")
            sel = torch.as_tensor(list(indices_in_batch), dtype=torch.long,
                                  device=self.pos.device)
            pos, pos_pool = self.pos[sel], self.pos_pool[sel]
        else:
            pos = self.pos.expand(rows, -1, -1)
            pos_pool = self.pos_pool.expand(rows, -1)

        caller_device, caller_dtype = x.device, x.dtype
        latents = x.to(self.device, self.dtype)
        timestep = (t.to(self.device, torch.float32) * NUM_TRAIN_TIMESTEPS)

        if self.do_cfg:
            # [uncond (B), cond (B)] along dim 0; the chunk(2) below matches.
            hidden = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            embeds = torch.cat([self.neg.expand(rows, -1, -1), pos], dim=0)
            pooled = torch.cat([self.neg_pool.expand(rows, -1), pos_pool], dim=0)
        else:
            hidden, embeds, pooled = latents, pos, pos_pool

        out = self.transformer(
            hidden_states=hidden,
            timestep=timestep.to(self.dtype),
            encoder_hidden_states=embeds,
            pooled_projections=pooled,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if self.do_cfg:
            uncond, cond = out.chunk(2)
            out = uncond + self.guidance_scale * (cond - uncond)
        return out.to(torch.float32).to(caller_device, caller_dtype)

    @torch.no_grad()
    def decode_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, C, h, w)`` latents -> ``(B, 3, H, W)`` float32 pixels in [-1, 1].

        The raw VAE output, before the image processor's rescale, so
        :func:`to_uint8` applies to it directly. Chunk at the call site: the
        decode peaks higher than the transformer does at 1024px.
        """
        vae = self.pipe.vae
        lat = latents.to(vae.device, vae.dtype)
        lat = lat / vae.config.scaling_factor + vae.config.shift_factor
        img = vae.decode(lat, return_dict=False)[0]
        return img.to(torch.float32).to(latents.device)


class SD3ChurnTarget(TargetTransition):
    """``m^q``: the mean of one churn reverse step over latents.

    ``forward_batch > 0`` splits each call into chunks of that many latents.
    The velocity is pointwise in the batch, so this is exact -- and it matters
    far more here than for EDM, because CFG doubles every forward: a round
    asking for ``sample_batch x |I|`` latents actually pushes twice that
    through the transformer.
    """

    def __init__(
        self,
        denoiser: SD3Denoiser,
        sigmas: torch.Tensor,
        eps: float,
        *,
        s_noise: float = 1.0,
        step_offset: int = 0,
        forward_batch: int = 0,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.sigmas = sigmas.to(torch.float64)
        self.eps = float(eps)
        self.s_noise = float(s_noise)
        self.step_offset = int(step_offset)
        self.forward_batch = int(forward_batch)

    def kernel(
        self,
        x: torch.Tensor,
        steps: Sequence[int],
        indices_in_batch: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(mean, std)`` of one reverse step, in *absolute* step indices."""
        idx = torch.as_tensor(list(steps), dtype=torch.long)
        wide = torch.promote_types(x.dtype, torch.float32)
        sigma = self.sigmas[idx].to(x.device, wide)
        sigma_next = self.sigmas[idx + 1].to(x.device, wide)
        dt = sigma_next - sigma  # < 0
        expand = lambda s: s.view(-1, *([1] * (x.dim() - 1)))  # noqa: E731

        v = self._velocity(x, sigma, indices_in_batch)
        det_mean = x + expand(dt) * v

        g2 = 2.0 * sigma / (1.0 - sigma).clamp_min(_SIGMA_GUARD)
        score = -(x + expand(1.0 - sigma) * v) / expand(sigma.clamp_min(_SIGMA_GUARD))
        churn_mean = x + expand(dt) * (v - 0.5 * self.eps**2 * expand(g2) * score)
        churn_std = self.eps * torch.sqrt(g2.clamp_min(0.0)) * torch.sqrt(-dt) * self.s_noise

        active = (
            (self.eps > 0.0)
            & (sigma > _SIGMA_GUARD)
            & (sigma < 1.0 - _SIGMA_GUARD)
            & (sigma_next > 0.0)
        )
        mean = torch.where(expand(active), churn_mean, det_mean)
        std = torch.where(active, churn_std, torch.zeros_like(churn_std))
        return mean, std

    def _velocity(self, x, sigma, indices_in_batch) -> torch.Tensor:
        n = x.shape[0]
        fb = self.forward_batch
        if fb <= 0 or n <= fb:
            return self.denoiser.velocity(x, sigma, indices_in_batch)
        idx = None if indices_in_batch is None else list(indices_in_batch)
        return torch.cat([
            self.denoiser.velocity(
                x[i : i + fb], sigma[i : i + fb],
                None if idx is None else idx[i : i + fb],
            )
            for i in range(0, n, fb)
        ])

    def means(self, indices_in_batch, states, steps):
        return self.kernel(
            states, [s + self.step_offset for s in steps], indices_in_batch
        )[0]


@dataclass(frozen=True)
class Setting:
    """One fully-specified latent-space configuration."""

    target: SD3ChurnTarget
    schedule: NoiseSchedule
    state_shape: Tuple[int, int, int]
    num_steps: int
    """Speculative steps -- ``total_steps`` minus the deterministic endpoints."""
    total_steps: int
    deterministic_steps: Tuple[int, ...]
    sigmas: torch.Tensor

    def initial_state(self, generator: torch.Generator, *, device="cpu") -> torch.Tensor:
        """Pure noise at ``sigma = 1``: the start of the *full* trajectory."""
        return torch.randn(
            self.state_shape, generator=generator, device=device, dtype=torch.float32
        )


def build(
    denoiser: SD3Denoiser,
    *,
    num_steps: int = 28,
    eps: float = 0.25,
    shift: float = DEFAULT_SHIFT,
    s_noise: float = 1.0,
    forward_batch: int = 0,
) -> Setting:
    """Assemble the target, the schedule, and the deterministic-endpoint bookkeeping."""
    if s_noise <= 0.0:
        # Caught here rather than left to the bookkeeping below, which would
        # mislead. At s_noise=0 every std is zero, `leading` walks the whole
        # grid, and the run dies naming *eps* -- which was fine. Negative values
        # are worse: nothing is exactly zero, so the endpoints look normal and a
        # negative std reaches TabulatedSchedule, to be refused rounds later,
        # after the checkpoint has loaded and generation has started.
        raise ValueError(
            f"s_noise must be > 0; got {s_noise}. It scales the transition std, "
            "so s_noise=0 makes every step deterministic and a negative value "
            "is not a std. To turn churn off, use eps=0."
        )
    sigmas = sigma_grid(num_steps, shift)
    std = churn_std_grid(sigmas, eps, s_noise=s_noise)
    zero = tuple(int(n) for n in torch.nonzero(std == 0.0).flatten())

    leading = 0
    while leading in zero:
        leading += 1
    trailing = 0
    while (num_steps - 1 - trailing) in zero:
        trailing += 1
    interior = [n for n in zero if leading <= n < num_steps - trailing]
    if interior:
        raise ValueError(
            f"transition std is zero at interior steps {interior}; speculation "
            "cannot span a deterministic step (Remark 3)"
        )

    speculative = num_steps - leading - trailing
    if speculative < 1:
        raise ValueError(
            f"no stochastic steps at eps={eps}, num_steps={num_steps}: every step "
            "is a deterministic Euler step, so there is nothing to speculate"
        )
    target = SD3ChurnTarget(
        denoiser, sigmas, eps, s_noise=s_noise, step_offset=leading,
        forward_batch=forward_batch,
    )
    return Setting(
        target=target,
        schedule=TabulatedSchedule(std[leading : leading + speculative].tolist()),
        state_shape=denoiser.state_shape,
        num_steps=speculative,
        total_steps=num_steps,
        deterministic_steps=zero,
        sigmas=sigmas,
    )


def euler_steps(target, y, steps, generator) -> torch.Tensor:
    """Apply the full kernel at each *absolute* step to a stack of latents."""
    for n in steps:
        rows = (int(n),) * y.shape[0]
        # The endpoints run before compaction, so the stack is the whole batch
        # in image order: entry i is image i.
        mean, std = target.kernel(y, rows, range(y.shape[0]))
        y = mean
        if float(std.max()) > 0.0:
            noise = torch.randn(y.shape, generator=generator, device=y.device,
                                dtype=y.dtype)
            y = y + std.view(-1, *([1] * (y.dim() - 1))) * noise
    return y


def sample_trajectory(setting: Setting, sampler, y0, *, rng, generator):
    """Full ``T``-step trajectory: Euler prologue, speculative middle, Euler epilogue.

    Returns **latents**, not pixels -- decoding is the caller's, because the VAE
    peaks higher than the transformer and wants its own chunking.
    """
    target, T = setting.target, setting.total_steps
    lo = target.step_offset
    hi = lo + setting.num_steps
    single = y0.dim() == len(setting.state_shape)

    y = euler_steps(target, y0[None] if single else y0, range(0, lo), generator)
    result = sampler.sample(y[0] if single else y, rng=rng)
    out = result.sample[None] if single else result.samples
    out = euler_steps(target, out, range(hi, T), generator)
    return (out[0] if single else out), result


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """``(..., 3, H, W)`` in [-1, 1] -> uint8, matching the pixel adapter."""
    return ((x.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
