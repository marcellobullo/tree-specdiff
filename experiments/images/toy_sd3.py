"""A closed-form stand-in for an SD3.5 pipeline, so the wiring is testable on a laptop.

SD3.5-medium is 16 GiB of weights and wants a GPU. None of that is needed to
check the parts of the adapter that can actually be wrong: the prompt table and
its two levels of indexing, the CFG stack and its ``chunk(2)``, the timestep
scaling, the churn kernel, the VAE decode, and the sampler plumbing.

So this supplies a pipeline with the *interface* ``SD3Denoiser`` uses, whose
transformer is exact. Take each prompt to name a constant latent ``mu``. Under
the linear interpolant ``x_t = (1 - t) mu + t xi`` the velocity is available in
closed form,

    v = E[xi - mu | x_t] = (x_t - mu) / t,

so a finished trajectory lands on that prompt's ``mu``. That turns "did image
3's caption reach image 3's entries?" into an assertion -- the failure this
whole prompt-table design exists to prevent, and one that would otherwise
produce plausible images of the wrong caption.

Under guidance the target moves predictably rather than vanishing: with
``v_u = (x - a)/t`` and ``v_c = (x - b)/t``, the guided velocity is
``(x - (a + s(b - a)))/t``, so :func:`guided_mu` gives the value a test should
expect at any guidance scale.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import torch

__all__ = ["ToySD3Pipeline", "prompt_mu", "guided_mu"]

NEGATIVE_MU = -0.5      # what the empty negative prompt names


def prompt_mu(prompt: str) -> float:
    """The constant latent a caption names. Deterministic, well separated."""
    if prompt == "":
        return NEGATIVE_MU
    h = hashlib.sha256(prompt.encode()).digest()
    return round(-0.8 + 1.6 * (int.from_bytes(h[:4], "big") / 2**32), 4)


def guided_mu(prompt: str, guidance_scale: float, negative: str = "") -> float:
    """Where a trajectory lands once classifier-free guidance is applied."""
    a, b = prompt_mu(negative), prompt_mu(prompt)
    return a + guidance_scale * (b - a)


class _Config:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _ToyTransformer:
    """``v = (x - mu) / t``, with ``mu`` read out of the pooled projection.

    Reads its conditioning exactly where the real transformer does, so the CFG
    stack and the ``chunk(2)`` split are genuinely exercised: each of the ``2B``
    rows carries its own pooled vector and gets its own ``mu``.
    """

    def __init__(self, in_channels: int, patch_size: int):
        self.config = _Config(in_channels=in_channels, patch_size=patch_size)

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states,
                 pooled_projections, joint_attention_kwargs=None, return_dict=False):
        x = hidden_states.to(torch.float32)
        # The adapter multiplies the interpolant coefficient by 1000 on the way
        # in; undo it here, which is what makes a scaling slip visible.
        t = (timestep.to(torch.float32) / 1000.0).clamp_min(1e-4)
        mu = pooled_projections.to(torch.float32)[:, 0]
        shape = (-1, *([1] * (x.dim() - 1)))
        v = (x - mu.view(shape)) / t.view(shape)
        return (v.to(hidden_states.dtype),)


class _ToyVAE:
    """Latents -> pixels: channel mean, upsampled by the scale factor."""

    def __init__(self, scale_factor: int, device, dtype):
        self.config = _Config(scaling_factor=1.5, shift_factor=0.1)
        self.device, self.dtype = device, dtype
        self._scale = scale_factor

    def decode(self, latents, return_dict=False):
        img = latents.to(torch.float32).mean(dim=1, keepdim=True)
        img = img.repeat_interleave(self._scale, dim=-1).repeat_interleave(self._scale, dim=-2)
        return (img.repeat(1, 3, 1, 1).to(self.dtype),)


class ToySD3Pipeline:
    """``StableDiffusion3Pipeline``'s interface, backed by an exact model."""

    def __init__(
        self,
        *,
        resolution_px: int = 64,
        in_channels: int = 4,
        patch_size: int = 2,
        vae_scale_factor: int = 8,
        seq_len: int = 8,
        embed_dim: int = 16,
        pooled_dim: int = 8,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        shift: float = 3.0,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.vae_scale_factor = vae_scale_factor
        self.transformer = _ToyTransformer(in_channels, patch_size)
        self.vae = _ToyVAE(vae_scale_factor, self.device, dtype)
        self.scheduler = _Config(config={"shift": shift, "num_train_timesteps": 1000})
        # Something for free_text_encoders() to drop, so that path is exercised.
        self.text_encoder = object()
        self.text_encoder_2 = object()
        self.text_encoder_3 = object()
        self._seq, self._embed, self._pooled = seq_len, embed_dim, pooled_dim

    def encode_prompt(self, *, prompt, prompt_2=None, prompt_3=None,
                      negative_prompt="", do_classifier_free_guidance=True,
                      device=None, num_images_per_prompt=1):
        if self.text_encoder is None:
            raise RuntimeError("text encoders were freed; cannot encode more prompts")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        n = len(prompts)

        def table(values: Sequence[float]):
            pooled = torch.zeros((len(values), self._pooled), dtype=torch.float32)
            pooled[:, 0] = torch.tensor(values, dtype=torch.float32)
            embeds = pooled[:, None, :].expand(-1, self._seq, self._pooled)
            embeds = torch.nn.functional.pad(
                embeds, (0, self._embed - self._pooled)
            ).contiguous()
            return embeds.to(self.device), pooled.to(self.device)

        pos, pos_pool = table([prompt_mu(p) for p in prompts])
        if not do_classifier_free_guidance:
            return pos, None, pos_pool, None
        neg, neg_pool = table([prompt_mu(negative_prompt)] * n)
        return pos, neg, pos_pool, neg_pool
