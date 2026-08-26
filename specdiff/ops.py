"""Minimal array-backend shim.

This module contains the array operations used by the sampler and verifiers.
Supporting another framework, such as JAX or MLX, requires a :class:`Backend`
subclass and registration.

State arrays are always *stacks*: shape ``(num_nodes, *state_shape)``, where
``state_shape`` is whatever the diffusion state is (``(d,)``, ``(C, H, W)``,
``(16, 64, 64)`` for an SD3 latent, ...). Node indices are plain Python ints,
step indices are plain Python ints; nothing framework-specific crosses the
public API except the state arrays themselves.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Sequence

Array = Any  # np.ndarray | torch.Tensor


class Backend(ABC):
    """The complete set of array operations the library depends on."""

    name: str

    @abstractmethod
    def zeros_stack(self, n: int, ref: Array) -> Array:
        """``(n, *ref.shape)`` zeros with ref's dtype/device."""

    @abstractmethod
    def randn_stack(self, n: int, ref: Array, rng: Any = None) -> Array:
        """``(n, *ref.shape)`` standard normal draws.

        Must reject a non-floating ``ref``: casting a standard normal to an
        integer dtype truncates every draw toward zero and invalidates the
        sample. Use
        :meth:`check_state_dtype`.
        """

    @abstractmethod
    def uniform(self, rng: Any = None) -> float:
        """A single scalar draw from Uniform[0, 1)."""

    @abstractmethod
    def make_rng(self, seed: Any = None, ref: Array = None) -> Any:
        """A fresh generator this backend's ``randn_stack``/``uniform`` accept.

        Exists so that code needing reproducible draws -- notably
        :func:`specdiff.testing.check_exactness` -- can seed a stream without
        knowing which framework is in play. ``ref``, when given, supplies the
        device the generator must live on.
        """

    @abstractmethod
    def finfo_eps(self, ref: Array) -> float:
        """Machine epsilon of ``ref``'s dtype.

        Exposed for rules that want a dtype-aware tolerance of their own.
        Note that the *degeneracy* tolerance is deliberately **not** one of
        them -- see :data:`specdiff.verifiers.rank1.DEFAULT_DEGENERATE_TOL`
        for the rationale behind the constant threshold.
        """

    @abstractmethod
    def is_floating(self, ref: Array) -> bool:
        """True if ``ref`` has a floating-point dtype."""

    def check_state_dtype(self, ref: Array, where: str = "state") -> None:
        """Raise unless ``ref`` is floating point.

        Diffusion states represent real-valued vectors. Integer arrays truncate
        noise draws and mean updates, invalidating the trajectory.
        """
        if not self.is_floating(ref):
            raise TypeError(
                f"{where} has non-floating dtype {getattr(ref, 'dtype', type(ref))!r}. "
                "Diffusion states must be floating point: an integer array truncates "
                "every Gaussian draw to zero and the sampler would return a silently "
                "wrong trajectory. Cast with e.g. `init.astype(float)`."
            )

    @abstractmethod
    def take(self, stack: Array, indices: Sequence[int]) -> Array:
        """Gather rows ``stack[indices]`` (copy)."""

    @abstractmethod
    def put(self, stack: Array, indices: Sequence[int], values: Array) -> None:
        """In-place ``stack[indices] = values``."""

    @abstractmethod
    def repeat_rows(self, stack: Array, counts: Sequence[int]) -> Array:
        """Repeat row ``i`` of ``stack`` ``counts[i]`` times, in order."""

    @abstractmethod
    def stack_rows(self, arrays: Sequence[Array]) -> Array:
        """Stack single states into ``(len(arrays), *state_shape)``."""

    @abstractmethod
    def scale_rows(self, stack: Array, scalars: Sequence[float]) -> Array:
        """Multiply row ``i`` by ``scalars[i]``.

        Needed once trajectories in a batch sit at different steps and so no
        longer share a single ``sigma``.
        """

    @abstractmethod
    def group_rows(self, stack: Array, group_size: int) -> Array:
        """``(batch * G, *shape) -> (batch, G, *shape)``, preserving order."""

    @abstractmethod
    def norm(self, x: Array) -> float:
        """Euclidean norm of a single state, as a Python float.

        Must reduce in **at least float32**, whatever dtype the state carries.
        A float16 reduction squares its inputs in float16, whose exponent range
        cannot hold the result: it underflows to exactly ``0`` below ``~2.4e-4``
        and overflows to ``inf`` above ``~256``, both well inside the ordinary
        range of a state. Since ``norm`` feeds ``Rank1Frame.delta``, a zero
        there is not a small error -- it makes every frame look degenerate and
        every coupling accept unconditionally.
        """

    @abstractmethod
    def dot(self, x: Array, y: Array) -> float:
        """Flat inner product of two single states, as a Python float.

        Same widening requirement as :meth:`norm`; this one feeds the projected
        coordinate ``S`` in :meth:`Rank1Frame.project`.
        """

    @abstractmethod
    def copy(self, x: Array) -> Array: ...

    @abstractmethod
    def is_finite(self, x: Array) -> bool: ...

    @abstractmethod
    def allclose(self, x: Array, y: Array) -> bool: ...

    def numel(self, x: Array) -> int:
        n = 1
        for s in tuple(x.shape):
            n *= int(s)
        return n


class NumpyBackend(Backend):
    name = "numpy"

    def __init__(self) -> None:
        import numpy as np

        self._np = np
        self._default_rng = np.random.default_rng()

    def _rng(self, rng):
        return self._default_rng if rng is None else rng

    def zeros_stack(self, n, ref):
        return self._np.zeros((n, *ref.shape), dtype=ref.dtype)

    def randn_stack(self, n, ref, rng=None):
        self.check_state_dtype(ref)
        out = self._rng(rng).standard_normal((n, *ref.shape))
        return out.astype(ref.dtype, copy=False)

    def uniform(self, rng=None):
        return float(self._rng(rng).random())

    def make_rng(self, seed=None, ref=None):
        return self._np.random.default_rng(seed)

    def finfo_eps(self, ref):
        return float(self._np.finfo(ref.dtype).eps)

    def is_floating(self, ref):
        return bool(self._np.issubdtype(ref.dtype, self._np.floating))

    def take(self, stack, indices):
        return stack[list(indices)]

    def put(self, stack, indices, values):
        stack[list(indices)] = values

    def repeat_rows(self, stack, counts):
        return self._np.repeat(stack, self._np.asarray(list(counts)), axis=0)

    def stack_rows(self, arrays):
        return self._np.stack(list(arrays), axis=0)

    def scale_rows(self, stack, scalars):
        shape = (len(stack),) + (1,) * (stack.ndim - 1)
        return stack * self._np.asarray(list(scalars), dtype=stack.dtype).reshape(shape)

    def group_rows(self, stack, group_size):
        return stack.reshape((-1, group_size) + stack.shape[1:])

    def _wide(self, x):
        """Flatten, and promote to at least float32 before reducing.

        See :meth:`Backend.norm`: a float16 reduction squares its inputs in
        float16, whose 5-bit exponent cannot hold the result. Promotion is a
        no-op for float32 and float64.
        """
        flat = x.reshape(-1)
        return flat.astype(self._np.promote_types(flat.dtype, self._np.float32), copy=False)

    def norm(self, x):
        return float(self._np.linalg.norm(self._wide(x)))

    def dot(self, x, y):
        return float(self._np.dot(self._wide(x), self._wide(y)))

    def copy(self, x):
        return x.copy()

    def is_finite(self, x):
        return bool(self._np.isfinite(x).all())

    def allclose(self, x, y):
        return bool(self._np.allclose(x, y))


class TorchBackend(Backend):
    name = "torch"

    def __init__(self) -> None:
        import torch

        self._torch = torch

    def zeros_stack(self, n, ref):
        return self._torch.zeros((n, *ref.shape), dtype=ref.dtype, device=ref.device)

    def randn_stack(self, n, ref, rng=None):
        self.check_state_dtype(ref)
        return self._torch.randn(
            (n, *ref.shape), dtype=ref.dtype, device=ref.device, generator=rng
        )

    def uniform(self, rng=None):
        # The draw has to happen on the generator's own device: `torch.rand`
        # otherwise allocates on the default device (cpu) and refuses a
        # generator from anywhere else. A run on cuda hands this a cuda
        # generator, so omitting `device` fails at the first acceptance test.
        device = None if rng is None else rng.device
        return float(self._torch.rand((), generator=rng, device=device).item())

    def make_rng(self, seed=None, ref=None):
        gen = self._torch.Generator(device="cpu" if ref is None else ref.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        return gen

    def finfo_eps(self, ref):
        return float(self._torch.finfo(ref.dtype).eps)

    def is_floating(self, ref):
        return bool(ref.dtype.is_floating_point)

    def take(self, stack, indices):
        idx = self._torch.as_tensor(list(indices), dtype=self._torch.long, device=stack.device)
        return stack.index_select(0, idx)

    def put(self, stack, indices, values):
        idx = self._torch.as_tensor(list(indices), dtype=self._torch.long, device=stack.device)
        stack[idx] = values

    def repeat_rows(self, stack, counts):
        c = self._torch.as_tensor(list(counts), dtype=self._torch.long, device=stack.device)
        return self._torch.repeat_interleave(stack, c, dim=0)

    def stack_rows(self, arrays):
        return self._torch.stack(list(arrays), dim=0)

    def scale_rows(self, stack, scalars):
        shape = (stack.shape[0],) + (1,) * (stack.dim() - 1)
        s = self._torch.as_tensor(
            list(scalars), dtype=stack.dtype, device=stack.device
        ).reshape(shape)
        return stack * s

    def group_rows(self, stack, group_size):
        return stack.reshape((-1, group_size) + tuple(stack.shape[1:]))

    def _wide(self, x):
        """Flatten, and promote to at least float32 before reducing.

        `torch.linalg.vector_norm` already widens internally, but `torch.dot`
        does not: a float16 dot underflows to zero below ~1e-4 per term.
        Promoting to float32 rather than float64 keeps this valid on devices
        without double support (MPS); float32 and float64 are unchanged.
        """
        flat = x.reshape(-1)
        return flat.to(self._torch.promote_types(flat.dtype, self._torch.float32))

    def norm(self, x):
        return float(self._torch.linalg.vector_norm(self._wide(x)).item())

    def dot(self, x, y):
        return float(self._torch.dot(self._wide(x), self._wide(y)).item())

    def copy(self, x):
        return x.clone()

    def is_finite(self, x):
        return bool(self._torch.isfinite(x).all().item())

    def allclose(self, x, y):
        return bool(self._torch.allclose(x, y))


_CACHE: dict[str, Backend] = {}


def resolve_backend(reference: Array) -> Backend:
    """Pick a backend from the type of an example state array."""
    module = type(reference).__module__.split(".")[0]
    key = "torch" if module == "torch" else "numpy" if module == "numpy" else module
    if key not in ("torch", "numpy"):
        raise TypeError(
            f"No backend registered for array type {type(reference)!r}. "
            "Subclass specdiff.ops.Backend and pass it to the sampler explicitly."
        )
    if key not in _CACHE:
        _CACHE[key] = TorchBackend() if key == "torch" else NumpyBackend()
    return _CACHE[key]


def default_state(dim: int) -> Array:
    """A zero state of shape ``(dim,)`` on the default backend.

    Only used to give :func:`specdiff.testing.check_exactness` a stand-in when
    the caller does not supply one. Lives here so that NumPy stays confined to
    this module.
    """
    import numpy as np

    return np.zeros(int(dim))


def standard_normal_cdf(x: float) -> float:
    """Phi(x). Kept here so verifiers/tests need no SciPy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def standard_normal_sf(x: float) -> float:
    """1 - Phi(x), the survival function used throughout the paper."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))
