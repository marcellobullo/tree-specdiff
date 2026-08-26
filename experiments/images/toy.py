"""Closed-form EDM checkpoint substitute for CPU integration tests.

The real checkpoints are hundreds of MB, need a NVlabs/edm checkout on
``sys.path``, and generally require a GPU. This substitute tests the change of
variables, churn kernel, endpoint bookkeeping, and sampler integration without
those dependencies.

The module supplies a network with the ``EDMPrecond`` interface whose
denoiser is exact. Take the data distribution to be a per-class isotropic
Gaussian ``x0 ~ N(mu_c, s^2 I)``. Under EDM's additive-noise convention
``y = x0 + sigma xi`` the posterior mean is available in closed form,

    D(y; sigma) = mu + s^2 / (s^2 + sigma^2) (y - mu),

and under the interpolant ``x_t = (1 - t) x0 + t xi`` so is the velocity,

    E[x0 | x_t] = mu + (1 - t) s^2 / ((1 - t)^2 s^2 + t^2) (x_t - (1 - t) mu)
    v           = (x_t - E[x0 | x_t]) / t.

:func:`analytic_velocity` computes the second directly. It is *not* derived from
the first, so comparing them validates :meth:`EDMDenoiser.velocity`
independently.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["GaussianEDMPrecond", "analytic_velocity"]


class GaussianEDMPrecond(torch.nn.Module):
    """``EDMPrecond``'s interface, backed by an exact Gaussian denoiser.

    Attribute names (``img_resolution``, ``img_channels``, ``label_dim``) and
    the ``(x, sigma, class_labels=...)`` call are EDM's, so
    :class:`~experiments.images.models.EDMDenoiser` cannot tell this apart from
    a real checkpoint.
    """

    def __init__(
        self,
        img_resolution: int = 8,
        img_channels: int = 3,
        label_dim: int = 0,
        *,
        data_std: float = 0.5,
        mean_scale: float = 0.3,
        seed: int = 20260823,
    ) -> None:
        super().__init__()
        self.img_resolution = int(img_resolution)
        self.img_channels = int(img_channels)
        self.label_dim = int(label_dim)
        self.data_std = float(data_std)
        shape = (max(label_dim, 1), img_channels, img_resolution, img_resolution)
        gen = torch.Generator().manual_seed(seed)
        # A buffer, so `.to(device, dtype)` moves it exactly as the weights of a
        # real checkpoint would.
        self.register_buffer(
            "class_means", mean_scale * torch.randn(shape, generator=gen)
        )

    def _mu(self, class_labels, rows: int, device, dtype) -> torch.Tensor:
        if self.label_dim == 0 or class_labels is None:
            return self.class_means[0].to(device, dtype).expand(rows, -1, -1, -1)
        # EDM passes one-hot rows; a matmul selects (and would blend, which no
        # caller does, but is the honest reading of the argument).
        flat = self.class_means.reshape(self.label_dim, -1).to(device, dtype)
        return (class_labels.to(device, dtype) @ flat).reshape(
            rows, self.img_channels, self.img_resolution, self.img_resolution
        )

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor, class_labels=None
    ) -> torch.Tensor:
        """``D(x; sigma) = E[x0 | x = x0 + sigma xi]``."""
        mu = self._mu(class_labels, x.shape[0], x.device, x.dtype)
        s2 = self.data_std**2
        gain = s2 / (s2 + sigma.to(x.device, x.dtype) ** 2)
        return mu + gain.view(-1, *([1] * (x.dim() - 1))) * (x - mu)


def analytic_velocity(
    net: GaussianEDMPrecond,
    x: torch.Tensor,
    t: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``v = E[xi - x0 | x_t]`` on the interpolant, computed without the denoiser.

    ``labels`` is one class per row, matching :meth:`EDMDenoiser.velocity`.
    """
    one_hot = None
    if net.label_dim and labels is not None:
        one_hot = torch.nn.functional.one_hot(
            torch.as_tensor(labels, dtype=torch.long), net.label_dim
        ).to(x.dtype)
    mu = net._mu(one_hot, x.shape[0], x.device, x.dtype)
    t4 = t.view(-1, *([1] * (x.dim() - 1))).to(x.device, x.dtype)
    s2 = net.data_std**2
    gain = (1.0 - t4) * s2 / ((1.0 - t4) ** 2 * s2 + t4**2)
    x0_hat = mu + gain * (x - (1.0 - t4) * mu)
    return (x - x0_hat) / t4
