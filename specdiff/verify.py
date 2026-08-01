"""The pluggable verification rule.

``Verify`` is the one component Algorithm 3 leaves abstract, and the paper is
unusually precise about its contract, so the base class states it as an
invariant rather than a docstring aspiration:

1.  ``result.state`` is an **exact** sample from ``N(mu_q, sigma^2 I)``.
2.  If ``result.accepted``, then ``result.state`` *is* one of the children.
3.  Otherwise ``result.state`` came from the normalised residual.

Point 1 is the reason the whole approach is worth anything, and it is not
checkable per-call. Point 2 is checkable and :class:`CheckedVerifier` checks
it; point 1 gets a statistical test in :mod:`specdiff.testing`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type

from .ops import Backend, resolve_backend
from .types import BatchedVerifyRequest, BatchedVerifyResult, VerifyRequest, VerifyResult


class Verifier(ABC):
    """Base class for verification rules.

    Subclasses implement :meth:`verify` and, if they are restricted in the
    number of proposals they can couple, set :attr:`max_children`. RMC is a
    single-proposal maximal coupling, so it sets ``max_children = 1`` and the
    sampler will refuse at construction time to run it on a branching tree
    rather than silently ignoring siblings.
    """

    name: str = "verifier"
    max_children: Optional[int] = None
    """Largest ``K`` this rule supports, or ``None`` for unbounded."""

    @abstractmethod
    def verify(self, request: VerifyRequest) -> VerifyResult: ...

    def verify_batch(self, request: BatchedVerifyRequest) -> BatchedVerifyResult:
        """Verify one node per trajectory, for the batched sampler.

        The default splits the batch and calls :meth:`verify` row by row, so
        every rule works under batching without being rewritten. Override it
        when the per-node work is worth vectorising -- for the paper's rules
        that means the scalar sweep over levels ``lambda_k`` runs on a
        ``(batch,)`` vector instead of in a Python loop, while the ``d``-dimensional
        projections and reconstructions become single batched ops.

        An override must stay row-independent: row ``j`` may only depend on
        ``request.row(j)``. Coupling rows to each other changes the joint law
        of the batch even if each marginal still looks right.
        """
        ops = resolve_backend(request.children)
        rows = [self.verify(request.row(j)) for j in range(request.batch_size)]
        return BatchedVerifyResult.from_rows(rows, ops)

    def __call__(self, request: VerifyRequest) -> VerifyResult:
        return self.verify(request)

    def supports(self, num_children: int) -> bool:
        return self.max_children is None or num_children <= self.max_children

    def check_topology(self, tree) -> None:
        """Raise if this rule cannot handle the tree it is about to be given.

        Called once by the sampler, so misconfiguration fails at build time.
        """
        if not self.supports(tree.branching):
            raise ValueError(
                f"{type(self).__name__} supports at most K={self.max_children} "
                f"proposals per node, but the draft tree has K={tree.branching}. "
                "Use DraftTree.chain(L) for single-proposal rules."
            )

    def reset(self) -> None:
        """Drop per-run state. Called at the start of each :meth:`sample`."""

    # Convenience for subclasses; they get a backend without importing ops.
    @staticmethod
    def backend_for(request: VerifyRequest) -> Backend:
        return resolve_backend(request.children)


class ResampleVerifier(Verifier):
    """Always rejects and draws a fresh ``Y ~ N(mu_q, sigma^2 I)``.

    Trivially exact and trivially useless: it commits one state per target
    call, so a run with it reproduces the standard sampler at ``1.00x``. That
    makes it the reference point for the driver -- if Algorithm 3 with this
    rule does not match a plain Euler-Maruyama loop in distribution, the bug is
    in the driver, not in the coupling.
    """

    name = "resample"

    def verify(self, request: VerifyRequest) -> VerifyResult:
        ops = self.backend_for(request)
        noise = ops.randn_stack(1, request.target_mean, request.rng)[0]
        return VerifyResult(
            state=request.target_mean + request.sigma * noise,
            accepted=False,
            proposals_examined=0,
        )


class CheckedVerifier(Verifier):
    """Wrapper that enforces the checkable half of the contract.

    Wrap a rule under development in this; the cost is a couple of comparisons
    per node. ``SpeculativeSampler(..., check_contract=True)`` does it for you.
    """

    def __init__(self, inner: Verifier) -> None:
        self.inner = inner
        self.name = f"checked({inner.name})"
        self.max_children = inner.max_children

    def reset(self) -> None:
        self.inner.reset()

    def check_topology(self, tree) -> None:
        self.inner.check_topology(tree)

    def _check_state(
        self,
        ops: Backend,
        state,
        accepted: bool,
        child_index,
        children,
        state_shape,
        num_children: int,
        where: str = "",
    ) -> None:
        """The checkable half of the contract, for one committed state.

        Shared by :meth:`verify` and :meth:`verify_batch` so that the two paths
        cannot drift apart: a rule that is rejected scalar must be rejected
        batched, with the same diagnostic.
        """
        if tuple(state.shape) != tuple(state_shape):
            raise ValueError(
                f"{self.inner.name}{where} returned state of shape {tuple(state.shape)}, "
                f"expected {tuple(state_shape)}"
            )
        if not ops.is_finite(state):
            raise FloatingPointError(f"{self.inner.name}{where} returned a non-finite state")
        if not accepted:
            return
        if not (0 <= child_index < num_children):
            raise IndexError(
                f"{self.inner.name}{where} accepted child_index={child_index} "
                f"with K={num_children}"
            )
        if not ops.allclose(state, children[child_index]):
            raise ValueError(
                f"{self.inner.name}{where} reported accepted=True but the returned state is "
                f"not child {child_index}. An accepted state must be the drafted state itself, "
                "otherwise the sampler descends into the wrong subtree."
            )

    def verify_batch(self, request: BatchedVerifyRequest) -> BatchedVerifyResult:
        result = self.inner.verify_batch(request)
        ops = resolve_backend(request.children)
        if not isinstance(result, BatchedVerifyResult):
            raise TypeError(
                f"{self.inner.name}.verify_batch returned {type(result)!r}, "
                "expected BatchedVerifyResult"
            )
        if int(result.states.shape[0]) != request.batch_size:
            raise ValueError(
                f"{self.inner.name}.verify_batch returned {int(result.states.shape[0])} rows "
                f"for a batch of {request.batch_size}"
            )
        # Row by row, with exactly the checks the scalar path applies. Delegating
        # to inner.verify_batch means CheckedVerifier.verify never runs on this
        # path, so the per-row checks have to be repeated here or they are lost.
        for j, (ok, idx) in enumerate(zip(result.accepted, result.child_index)):
            self._check_state(
                ops,
                result.states[j],
                ok,
                idx,
                request.children[j],
                request.state_shape,
                request.num_children,
                where=f" row {j}",
            )
        return result

    def verify(self, request: VerifyRequest) -> VerifyResult:
        result = self.inner.verify(request)
        ops = self.backend_for(request)
        if not isinstance(result, VerifyResult):
            raise TypeError(f"{self.inner.name} returned {type(result)!r}, expected VerifyResult")
        self._check_state(
            ops,
            result.state,
            result.accepted,
            result.child_index,
            request.children,
            request.state_shape,
            request.num_children,
        )
        return result


# ------------------------------------------------------------------ registry
_REGISTRY: Dict[str, Type[Verifier]] = {}


def register_verifier(name: str) -> Callable[[Type[Verifier]], Type[Verifier]]:
    """Class decorator making a rule addressable by name from config files."""

    def deco(cls: Type[Verifier]) -> Type[Verifier]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise KeyError(f"verifier {name!r} is already registered to {_REGISTRY[key]!r}")
        cls.name = name
        _REGISTRY[key] = cls
        return cls

    return deco


def create_verifier(name: str, **kwargs: Any) -> Verifier:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown verifier {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_verifiers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


register_verifier("resample")(ResampleVerifier)
