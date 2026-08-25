"""Stable Diffusion 3.5 behind specdiff's ``TargetTransition``, in latent space.

The pixel-space adapter is ``models.py``; this is its latent-space counterpart,
kept separate because the two differ in every way that matters operationally
even though the churn math is identical:

* states are VAE **latents** ``(16, px/8, px/8)``, not images, so a decode step
  stands between the sampler and anything that looks at pixels;
* conditioning is **text**, one caption per image, encoded once up front;
* classifier-free guidance **doubles every forward**, so the memory a round
  needs is twice what the node count suggests.

Unlike EDM there is no change of variables: SD3 is trained on the same linear
flow-matching interpolant specdiff samples, so the transformer's output *is* the
velocity. All this module does is guide it and wrap the churn step around it.

Prompts
-------
``prompt`` may be a list, one caption per image. Every caption is encoded once
into a **CPU-resident** table and selected per latent by index -- at seq=333
a single embedding is ~2.7 MB, so a 30k COCO set is 82 GB and cannot live on
the GPU. :meth:`SD3Denoiser.set_prompt_batch` uploads only the rows a batch
needs; ``indices_in_batch`` then selects within them.

That second indexing level is why the port is clean: specdiff already hands
every entry of a target call its ``indices_in_batch``, and those are already
batch-local, so nothing in the sampler had to learn about prompts. (The
reference implementation had to route the index through its class-label channel
for want of anywhere else to put it.)

Duplication, deliberately
-------------------------
``sigma_grid`` and ``churn_std_grid`` are copied from ``models.py`` rather than
imported, so the latent and pixel experiments stay independent files. That
invites drift, so ``tests/test_sd3.py`` asserts the two implementations agree
exactly; if you change one, that test tells you about the other.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.device = pipe.device
        self.dtype = pipe.dtype
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

        pos, pool = [], []
        with torch.no_grad():
            for i in range(0, len(prompts), encode_batch):
                p, n, pp, np_ = pipe.encode_prompt(
                    prompt=prompts[i : i + encode_batch],
                    prompt_2=None,
                    prompt_3=None,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=self.do_cfg,
                    device=self.device,
                    num_images_per_prompt=1,
                )
                if not pos:
                    # One negative for the whole run: encode_prompt hands it back
                    # duplicated per row, so keep one and expand at call time.
                    self.neg = None if n is None else n[:1].clone()
                    self.neg_pool = None if np_ is None else np_[:1].clone()
                pos.append(p.to("cpu"))
                pool.append(pp.to("cpu"))

        self._pos_table = torch.cat(pos)      # (P, seq, dim), on the CPU
        self._pool_table = torch.cat(pool)    # (P, pooled_dim), on the CPU
        if self.per_sample_prompts:
            self.pos = self.pos_pool = None   # uploaded per batch
        else:
            self.pos = self._pos_table.to(self.device)
            self.pos_pool = self._pool_table.to(self.device)

        if free_text_encoders:
            self.free_text_encoders()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "SD3Denoiser":
        """Build from a hub id or a local diffusers directory."""
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.to(device)
        return cls(pipe, **kwargs)

    def free_text_encoders(self) -> None:
        """Drop the text towers once every prompt is encoded.

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
