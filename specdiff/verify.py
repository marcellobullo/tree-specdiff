"""The pluggable verification rule.

Algorithm 3 leaves ``Verify`` abstract. The base class expresses the paper's
contract as three invariants:

1.  ``result.state`` is an **exact** sample from ``N(mu_q, sigma^2 I)``.
2.  If ``result.accepted``, then ``result.state`` *is* one of the children.
3.  Otherwise ``result.state`` came from the normalised residual.

Distributional exactness cannot be checked per call. :class:`CheckedVerifier`
enforces the structural invariants, while :mod:`specdiff.testing` provides a
statistical exactness test.
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
    sampler rejects incompatible branching trees during construction.
    """

    name: str = "verifier"
    max_children: Optional[int] = None
    """Largest ``K`` this rule supports, or ``None`` for unbounded."""

    @abstractmethod
    def verify(self, request: VerifyRequest) -> VerifyResult: ...

    def verify_batch(self, request: BatchedVerifyRequest) -> BatchedVerifyResult:
        """Verify one node per trajectory, for the batched sampler.

        The default splits the batch and calls :meth:`verify` row by row, so
        every rule supports batching without changes. Override it when the
        per-node work benefits from vectorization. For the paper's rules,
        that means the scalar sweep over levels ``lambda_k`` runs on a
        ``(batch,)`` vector instead of in a Python loop, while the ``d``-dimensional
        projections and reconstructions become single batched ops.

        An override must stay row-independent: row ``j`` may only depend on
        ``request.row(j)``. Coupling rows to each other changes the joint law
        of the batch even if each marginal still looks right.
        """
        ops = request.backend or resolve_backend(request.children)
        rows = [self.verify(request.row(j)) for j in range(request.batch_size)]
        return BatchedVerifyResult.from_rows(rows, ops)

    def __call__(self, request: VerifyRequest) -> VerifyResult:
        return self.verify(request)

    def supports(self, num_children: int) -> bool:
        return self.max_children is None or num_children <= self.max_children

    @property
    def requires_chain(self) -> bool:
        """Whether experiments must match their allocated tree with a chain."""
        return self.max_children == 1

    def matched_tree(self, tree, *, num_steps: int, match: str = "verification",
                     evaluate_leaves: bool = False):
        """Apply the experiment's budget policy using verifier capabilities."""
        if match not in ("verification", "budget"):
            raise ValueError("match must be 'verification' or 'budget'")
        if self.requires_chain:
            from .trees import DraftTree

            depth = (tree.budget if match == "budget" else
                     tree.verification_budget(evaluate_leaves=evaluate_leaves)
                     - int(evaluate_leaves))
            tree = DraftTree.chain(max(1, min(depth, num_steps)))
        self.check_topology(tree)
        return tree

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
        return request.backend or resolve_backend(request.children)


class ResampleVerifier(Verifier):
    """Reject every draft and draw ``Y ~ N(mu_q, sigma^2 I)``.

    This exact reference verifier commits one state per target call and
    reproduces the standard sampler at ``1.00x``. It is useful for validating
    the sampler independently of a coupling.
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

    Enable this wrapper with ``SpeculativeSampler(..., check_contract=True)``
    while developing a verifier. It adds a small number of comparisons per node.
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
        ops = request.backend or resolve_backend(request.children)
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
