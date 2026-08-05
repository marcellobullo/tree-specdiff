"""Speculative diffusion sampling over arbitrary draft trees.

Algorithm 3 of *Accelerating Diffusion Sampling via Speculative Draft Trees*.
The sampler knows nothing about the two things the paper varies -- the draft
topology and the verification rule -- so you supply those and nothing else.

    from specdiff import DraftTree, DelayedDriftProposal, SpeculativeSampler

    sampler = SpeculativeSampler(
        target=my_target,
        proposal=DelayedDriftProposal(my_target),
        schedule=my_schedule,
        tree=DraftTree.uniform(branching=4, lookahead=3),
        verifier=my_rule,
        num_steps=100,
    )
    result = sampler.sample(y0, rng=rng)
"""

from __future__ import annotations

from .batched import (
    BatchedDelayedDriftProposal,
    BatchedProposal,
    BatchedSpeculativeSampler,
    PerSlotProposal,
    StatelessBatchedProposal,
)
from .kernels import (
    ConstantSchedule,
    DelayedDriftProposal,
    IdentityProposal,
    MirrorProposal,
    NoiseSchedule,
    ProposalTransition,
    TabulatedSchedule,
    TargetTransition,
)
from .ops import Backend, resolve_backend
from .sampler import SpeculativeSampler, standard_sampler
from .testing import ExactnessReport, check_exactness
from .trees import ROOT, DraftTree
from .types import (
    BatchedRoundRecord,
    BatchedSamplingResult,
    BatchedVerifyRequest,
    BatchedVerifyResult,
    RoundRecord,
    SamplingResult,
    VerifyRequest,
    VerifyResult,
)
from .verify import (
    CheckedVerifier,
    ResampleVerifier,
    Verifier,
    available_verifiers,
    create_verifier,
    register_verifier,
)

__version__ = "0.1.0"

__all__ = [
    # samplers
    "SpeculativeSampler",
    "BatchedSpeculativeSampler",
    "standard_sampler",
    # topology
    "DraftTree",
    "ROOT",
    # models
    "TargetTransition",
    "ProposalTransition",
    "IdentityProposal",
    "MirrorProposal",
    "DelayedDriftProposal",
    "NoiseSchedule",
    "ConstantSchedule",
    "TabulatedSchedule",
    # batched proposals
    "BatchedProposal",
    "StatelessBatchedProposal",
    "BatchedDelayedDriftProposal",
    "PerSlotProposal",
    # the pluggable rule
    "Verifier",
    "CheckedVerifier",
    "ResampleVerifier",
    "register_verifier",
    "create_verifier",
    "available_verifiers",
    # contract types
    "VerifyRequest",
    "VerifyResult",
    "BatchedVerifyRequest",
    "BatchedVerifyResult",
    "RoundRecord",
    "SamplingResult",
    "BatchedRoundRecord",
    "BatchedSamplingResult",
    # backend + tooling
    "Backend",
    "resolve_backend",
    "check_exactness",
    "ExactnessReport",
    "__version__",
]
