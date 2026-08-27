"""Pretrained EDM checkpoints as a specdiff ``TargetTransition``.

Karras et al. (2022) publish CIFAR10 and FFHQ checkpoints
(https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/, CC BY-NC-SA 4.0) whose
``EDMPrecond`` network is a *denoiser*. specdiff wants eq. (24): a transition
mean ``m^q_n`` and a shared scale ``sigma_n``. The adapter applies two exact
changes of variables and requires no retraining.

1.  **Denoiser -> velocity.** EDM's ``D(y; s) = E[x0 | y = x0 + s n]`` lives on
    an additive-noise parameterisation; specdiff's trajectory lives on the
    linear interpolant ``x_t = (1 - t) x0 + t xi``. Dividing by ``(1 - t)``,

        x_t / (1 - t) = x0 + s xi,      s = t / (1 - t),

    which *is* EDM's convention, so ``D(x_t / (1 - t); s) = E[x0 | x_t]`` and

        v = E[xi - x0 | x_t] = (x_t - D) / t.

    Both endpoints are singular; ``EDMDenoiser`` clamps ``t`` to the range
    implied by ``[sigma_min, sigma_max]`` and uses the *same* clamped ``t`` for
    the network call and the division, so the approximation stays consistent.

2.  **Velocity -> transition.** One churn reverse step (De Bortoli et al. 2025,
    eqs. 35/37) integrated in decreasing sigma, ``dt = sigma_next - sigma < 0``::

        g^2(sigma) = 2 sigma / (1 - sigma)
        score      = -(x + (1 - sigma) v) / sigma
        mean       = x + dt (v - (1/2) eps^2 g^2 score)
        std        = eps sqrt(g^2) sqrt(-dt)

    ``mean`` is ``m^q_n``; ``std`` is ``sigma_n``, and it is *state-independent*,
    satisfying the shared-covariance condition required by the rank-1 reduction
    in Equations 8--11.
    So the schedule is a plain :class:`~specdiff.TabulatedSchedule`.

Deterministic endpoints
-----------------------
``std`` vanishes at two steps of every run: the first (``sigma = 1``, where
``g`` diverges) and the last (``sigma_next = 0``). Speculation is vacuous at
zero churn -- the two kernels become distinct point masses, TV distance 1 --
and ``NoiseSchedule`` refuses a non-positive scale outright (Remark 3). Like
``experiments/gm/models.py``, :func:`build` exposes only the stochastic steps
and reports the skipped ones explicitly. :func:`sample_trajectory` stitches
the deterministic prologue and epilogue back on: a ``T = 100`` run is 98
speculative steps plus 2 Euler steps that every sampler pays.

Conditioning
------------
Class labels are per image: ``ChurnKernelTarget`` holds a ``(batch_size,)``
tensor and gathers ``labels[indices_in_batch]`` for each entry of a call, so one
batch can mix classes. Set it per batch with :meth:`ChurnKernelTarget.set_class_labels`.

**EDM's conditional networks have no null class.** They were not trained for
classifier-free guidance, so there is no "unconditional" setting of a
``*-cond-*`` checkpoint: passing no label makes ``EDMPrecond`` fall back to a
zero embedding, which is not a trained null token and does not represent the
intended distribution. To sample the class marginal -- the usual
FID protocol, and what the reference implementation did -- draw a label per
image uniformly. Unconditional generation means an unconditional *checkpoint*
(``label_dim == 0``), not a conditional one with the label omitted. Both guards
are enforced: labels on an uncond checkpoint raise, and a missing label on a
cond checkpoint raises.

"""

from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from specdiff import NoiseSchedule, TabulatedSchedule, TargetTransition

__all__ = [
    "NUM_TRAIN_TIMESTEPS",
    "sigma_grid",
    "churn_std_grid",
    "EDMDenoiser",
    "ChurnKernelTarget",
    "Setting",
    "build",
    "euler_steps",
    "sample_trajectory",
    "to_uint8",
]

NUM_TRAIN_TIMESTEPS = 1000

# Guards on the 1/sigma and 1/(1 - sigma) singularities, matching
# ChurnFlowMatchEulerScheduler._eps_num in the reference implementation.
_SIGMA_GUARD = 1e-6


def sigma_grid(num_steps: int, shift: float = 1.0) -> torch.Tensor:
    """The ``(num_steps + 1,)`` sigma grid, terminal 0 included.

    Reproduces what ``diffusers.FlowMatchEulerDiscreteScheduler`` produces at
    ``shift`` (1.0 for EDM, 3.0 for SD3.5), so a run here and a run through the
    stock pipeline visit the same noise levels.
    """
    timesteps = torch.linspace(NUM_TRAIN_TIMESTEPS, 1.0, num_steps, dtype=torch.float64)
    sigmas = timesteps / NUM_TRAIN_TIMESTEPS
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)])


def churn_std_grid(
    sigmas: torch.Tensor, eps: float, *, s_noise: float = 1.0
) -> torch.Tensor:
    """Per-step transition std ``(num_steps,)``; 0 marks a deterministic step.

    State-independent by construction, which is what lets the schedule be a
    lookup table instead of a callback into the model.
    """
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


class EDMDenoiser:
    """A pretrained ``EDMPrecond`` exposed as a flow-matching velocity field.

    Parameters
    ----------
    net:
        A loaded ``EDMPrecond`` (see :meth:`from_pickle`), or anything with the
        same ``(x, sigma, class_labels=...)`` call and ``img_resolution`` /
        ``img_channels`` / ``label_dim`` attributes.
    sigma_min, sigma_max:
        Noise range the denoiser is trusted over. EDM's own sampling range,
        except ``sigma_min`` is lowered to 0.001 so a 100-step schedule (whose
        smallest ``t`` is 0.001) is not clamped at the low end.
    """

    def __init__(
        self,
        net,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        sigma_min: float = 0.001,
        sigma_max: float = 80.0,
    ) -> None:
        self.net = net.to(device=device, dtype=dtype).eval()
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_classes = int(getattr(net, "label_dim", 0))
        resolution = int(net.img_resolution)
        self.state_shape: Tuple[int, int, int] = (
            int(net.img_channels), resolution, resolution
        )
        # sigma = t / (1 - t)  <=>  t = sigma / (1 + sigma)
        self._t_min = sigma_min / (1.0 + sigma_min)
        self._t_max = sigma_max / (1.0 + sigma_max)

    @classmethod
    def from_pickle(
        cls,
        path: str,
        edm_repo: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        key: str = "ema",
        **kwargs,
    ) -> "EDMDenoiser":
        """Load an official EDM ``.pkl``; ``edm_repo`` is a NVlabs/edm checkout.

        Unpickling needs EDM's ``torch_utils`` and ``dnnlib`` importable (the
        class definitions are embedded via ``torch_utils.persistence``) and
        **executes code from the pickle**, so point ``edm_repo`` at a checkout
        you trust.
        """
        if edm_repo not in sys.path:
            sys.path.insert(0, edm_repo)
        with open(path, "rb") as f:
            net = pickle.load(f)[key]
        return cls(net, device=device, dtype=dtype, **kwargs)

    def _one_hot(self, labels: Optional[torch.Tensor]):
        """EDM takes one-hot vectors ``(rows, label_dim)``; None when uncond."""
        if self.num_classes == 0 or labels is None:
            return None
        idx = torch.as_tensor(labels, device=self.device, dtype=torch.long)
        return torch.nn.functional.one_hot(idx, self.num_classes).to(self.dtype)

    @torch.no_grad()
    def velocity(
        self, x: torch.Tensor, t: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``v = (x - D(x / (1 - t); t / (1 - t))) / t``, one row per element of ``t``.

        ``t`` is the interpolant coefficient -- the same quantity the schedule
        calls ``sigma`` -- not EDM's noise level.

        The *network* runs in ``self.dtype`` (bf16 is normal on a GPU); the
        change of variables around it runs in at least float32, promoted to the
        caller's dtype if that is wider. ``(x - D) / t`` is a cancellation
        divided by a small number -- at ``t = 0.01`` it amplifies the denoiser's
        error a hundredfold -- so a float64 trajectory must not be silently
        rounded to float32 on the way through, the same reason
        ``TorchBackend._wide`` promotes before reducing.
        """
        caller_device, caller_dtype = x.device, x.dtype
        wide = torch.promote_types(caller_dtype, torch.float32)
        x_in = x.to(self.device, wide)
        t = t.to(self.device, wide).clamp(self._t_min, self._t_max)
        t4 = t.view(-1, *([1] * (x_in.dim() - 1)))

        denoised = self.net(
            (x_in / (1.0 - t4)).to(self.dtype),
            (t / (1.0 - t)).to(self.dtype),
            class_labels=self._one_hot(labels),
        ).to(wide)

        return ((x_in - denoised) / t4).to(caller_device, caller_dtype)


class ChurnKernelTarget(TargetTransition):
    """``m^q``: the mean of one churn reverse step, at per-row step indices.

    Rows of a verification batch sit at different steps, and this kernel is
    vectorised over them, so ``means`` is a *single* network call -- no grouping
    loop of the kind ``examples/gaussian_mixture.py`` needs for a scalar-timestep
    model. EDM takes a per-sample sigma, so batching is free.

    ``forward_batch > 0`` splits that call into chunks of that many rows. The
    velocity is pointwise in the batch, so this is exact: it trades one large
    activation peak for several small ones, which is what decouples memory from
    the tree size (a ``(K, L)`` round asks for ``|I|`` rows at once). It does
    **not** inflate the NFE count -- :meth:`TargetTransition.__call__` counts one
    call per invocation regardless of what ``means`` does internally -- but it is
    not bit-reproducible across values, since cuDNN kernels are batch-size
    dependent. Keep it fixed across a comparison set.
    """

    def __init__(
        self,
        denoiser: EDMDenoiser,
        sigmas: torch.Tensor,
        eps: float,
        *,
        s_noise: float = 1.0,
        step_offset: int = 0,
        forward_batch: int = 0,
        class_labels: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.sigmas = sigmas.to(torch.float64)
        self.eps = float(eps)
        self.s_noise = float(s_noise)
        self.step_offset = int(step_offset)
        self.forward_batch = int(forward_batch)
        self.class_labels: Optional[torch.Tensor] = None
        self.set_class_labels(class_labels)

    def set_class_labels(self, labels: Optional[torch.Tensor]) -> None:
        """One class per image, for the batch about to be sampled, or ``None``.

        ``labels[b]`` is image ``b``'s class; entries of a call pick theirs up
        through ``indices_in_batch``. Call this once per batch, before
        sampling. ``None`` is only valid for an unconditional checkpoint -- see
        the module docstring on why a conditional one cannot be run without
        labels.
        """
        if labels is None:
            self.class_labels = None
            return
        if self.denoiser.num_classes == 0:
            raise ValueError(
                "class labels were given but this checkpoint is unconditional "
                "(label_dim = 0); load a *-cond-* .pkl or drop the labels"
            )
        labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
        if int(labels.max()) >= self.denoiser.num_classes or int(labels.min()) < 0:
            raise ValueError(
                f"class labels must lie in [0, {self.denoiser.num_classes}); got "
                f"[{int(labels.min())}, {int(labels.max())}]"
            )
        self.class_labels = labels

    def labels_for(self, indices_in_batch: Sequence[int]) -> Optional[torch.Tensor]:
        """The per-entry label vector for one call, or ``None`` when uncond."""
        if self.denoiser.num_classes == 0:
            return None
        if self.class_labels is None:
            raise ValueError(
                "this checkpoint is conditional (label_dim = "
                f"{self.denoiser.num_classes}) but no class labels are set. EDM's "
                "conditional networks have no null class, so there is no "
                "unconditional mode: call set_class_labels(...) -- drawing one "
                "label per image uniformly gives the class marginal -- or load an "
                "unconditional checkpoint"
            )
        return self.class_labels[list(indices_in_batch)]

    def kernel(
        self,
        x: torch.Tensor,
        steps: Sequence[int],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(mean, std)`` of one reverse step, in *absolute* step indices.

        The single source of truth for the step math, so the deterministic
        endpoints in :func:`sample_trajectory` and the speculative rounds cannot
        drift apart. ``labels`` is one class per row, or ``None``.
        """
        idx = torch.as_tensor(list(steps), dtype=torch.long)
        wide = torch.promote_types(x.dtype, torch.float32)
        sigma = self.sigmas[idx].to(x.device, wide)
        sigma_next = self.sigmas[idx + 1].to(x.device, wide)
        dt = sigma_next - sigma  # < 0
        expand = lambda s: s.view(-1, *([1] * (x.dim() - 1)))  # noqa: E731

        v = self._velocity(x, sigma, labels)
        det_mean = x + expand(dt) * v  # deterministic Euler fallback

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

    def _velocity(self, x, sigma, labels) -> torch.Tensor:
        n = x.shape[0]
        fb = self.forward_batch
        if fb <= 0 or n <= fb:
            return self.denoiser.velocity(x, sigma, labels)
        return torch.cat([
            self.denoiser.velocity(
                x[i : i + fb], sigma[i : i + fb],
                None if labels is None else labels[i : i + fb],
            )
            for i in range(0, n, fb)
        ])

    def means(self, indices_in_batch, states, steps):
        return self.kernel(
            states,
            [s + self.step_offset for s in steps],
            self.labels_for(indices_in_batch),
        )[0]


@dataclass(frozen=True)
class Setting:
    """One fully-specified image-generation configuration."""

    target: ChurnKernelTarget
    schedule: NoiseSchedule
    state_shape: Tuple[int, int, int]
    num_steps: int
    """Speculative steps -- ``total_steps`` minus the deterministic endpoints."""
    total_steps: int
    """``T`` as passed; the cost baseline a standard sampler pays."""
    deterministic_steps: Tuple[int, ...]
    """Absolute indices whose transition std is zero, in the full ``T`` grid."""
    sigmas: torch.Tensor
    """The ``(T + 1,)`` interpolant grid, terminal 0 included."""

    def initial_state(self, generator: torch.Generator, *, device="cpu") -> torch.Tensor:
        """Pure noise at ``sigma = 1``, i.e. the start of the *full* trajectory."""
        return torch.randn(
            self.state_shape, generator=generator, device=device, dtype=torch.float32
        )


def build(
    denoiser: EDMDenoiser,
    *,
    num_steps: int = 100,
    eps: float = 0.25,
    shift: float = 1.0,
    s_noise: float = 1.0,
    forward_batch: int = 0,
    class_labels: Optional[torch.Tensor] = None,
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
        # A zero in the middle would make the speculative window straddle a step
        # the schedule must refuse, and there is no sensible way to reindex
        # around it. Only reachable via churn_sigma_min/max windowing, which
        # this module deliberately does not expose.
        raise ValueError(
            f"transition std is zero at interior steps {interior}; speculation "
            "cannot span a deterministic step (Remark 3)"
        )

    target = ChurnKernelTarget(
        denoiser, 
        sigmas, 
        eps, 
        s_noise=s_noise, 
        step_offset=leading,
        forward_batch=forward_batch, 
        class_labels=class_labels,
    )
    
    speculative = num_steps - leading - trailing
    if speculative < 1:
        raise ValueError(
            f"no stochastic steps at eps={eps}, num_steps={num_steps}: every "
            "step is a deterministic Euler step, so there is nothing to speculate"
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


def euler_steps(
    target: "ChurnKernelTarget",
    y: torch.Tensor,
    steps,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply the full kernel at each *absolute* step in ``steps`` to a stack ``y``.

    Every row takes the same step, which is what the endpoints need: they sit
    outside the speculative window, where all trajectories are still in lockstep.
    Draws noise when the std is non-zero, so this is a correct sampler for any
    step, not only the deterministic ones.
    """
    for n in steps:
        rows = (int(n),) * y.shape[0]
        # The endpoints run before compaction, so the stack is the whole batch
        # in image order: entry i is image i.
        mean, std = target.kernel(y, rows, target.labels_for(range(y.shape[0])))
        y = mean
        if float(std.max()) > 0.0:
            noise = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)
            y = y + std.view(-1, *([1] * (y.dim() - 1))) * noise
    return y


def sample_trajectory(
    setting: "Setting",
    sampler,
    y0: torch.Tensor,
    *,
    rng,
    generator: torch.Generator,
    on_round=None,
):
    """Full ``T``-step trajectory: Euler prologue, speculative middle, Euler epilogue.

    ``sampler`` runs only the stochastic window; the endpoints carry zero
    transition noise and are plain Euler steps that every sampler -- speculative
    or not -- pays identically. Excluding them from the sampler is what keeps
    the NFE accounting honest: they are not what speculation is credited for,
    and both arms of a comparison pay exactly two of them.

    ``y0`` is ``(*state_shape)`` for :class:`~specdiff.SpeculativeSampler` or
    ``(batch, *state_shape)`` for :class:`~specdiff.BatchedSpeculativeSampler`;
    the rank decides, so the two samplers need no flag to tell apart.

    ``on_round``, when given, is forwarded to the batched sampler as its
    per-round progress hook. The Euler endpoints are not reported: they are two
    steps out of ``T``, and they run outside the sampler.

    Returns ``(final_state, result)`` with ``final_state`` the same rank as
    ``y0`` and ``result`` whatever the sampler returned.
    """
    target, T = setting.target, setting.total_steps
    lo = target.step_offset
    hi = lo + setting.num_steps
    single = y0.dim() == len(setting.state_shape)

    y = euler_steps(target, y0[None] if single else y0, range(0, lo), generator)
    # Only the batched sampler takes the hook; keep the call signature the one
    # SpeculativeSampler accepts when no reporting was asked for.
    hook = {} if on_round is None else {"on_round": on_round}
    result = sampler.sample(y[0] if single else y, rng=rng, **hook)
    out = result.sample[None] if single else result.samples
    out = euler_steps(target, out, range(hi, T), generator)
    return (out[0] if single else out), result


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """``(..., C, H, W)`` in [-1, 1] -> uint8, EDM's own quantisation."""
    return ((x.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
