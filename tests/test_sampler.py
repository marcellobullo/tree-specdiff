"""Sampler tests. Run with `python -m pytest tests` or `python tests/test_sampler.py`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdiff import (  # noqa: E402
    ConstantSchedule,
    DelayedDriftProposal,
    DraftTree,
    ResampleVerifier,
    SpeculativeSampler,
    TargetTransition,
    VerifyRequest,
    VerifyResult,
    Verifier,
    check_exactness,
    standard_sampler,
)
from specdiff.kernels import ProposalTransition  # noqa: E402


# --------------------------------------------------------------------- fixtures
class LinearGaussianTarget(TargetTransition):
    """m^q(y) = a * y. Exact terminal law is available in closed form."""

    def __init__(self, a: float = 0.9) -> None:
        super().__init__()
        self.a = a

    def means(self, states, steps):
        return self.a * states


class ExactProposal(ProposalTransition):
    """Same mean map as the target, but does not spend an NFE (test double)."""

    def __init__(self, a: float) -> None:
        self.a = a

    def means(self, states, steps):
        return self.a * states


class AcceptFirstVerifier(Verifier):
    """Exact only when the proposal equals the target (delta = 0, Remark 2)."""

    name = "accept-first"

    def verify(self, request: VerifyRequest) -> VerifyResult:
        return VerifyResult(state=request.child(0), accepted=True, child_index=0, proposals_examined=1)


class BrokenVerifier(Verifier):
    """Returns a draft uncorrected. Should be caught by check_exactness."""

    name = "broken"

    def verify(self, request):
        return VerifyResult(state=request.child(0), accepted=True, child_index=0)


# ------------------------------------------------------------------ tree algebra
def test_tree_budget_and_internals():
    for K in (1, 2, 3, 4):
        for L in (1, 2, 3):
            tree = DraftTree.uniform(K, L)
            assert tree.budget == sum(K**l for l in range(1, L + 1))  # eq. (12)
            assert len(tree.internal_nodes) == tree.budget // K  # eq. (26)
            assert tree.depth == L
            assert len(tree.layer(L)) == K**L
    assert DraftTree.chain(5).budget == 5
    assert DraftTree.largest_uniform(budget=155, branching=2).depth == 6  # 2+..+2^6 = 126
    widths = DraftTree.from_widths([3, 1, 2])
    assert widths.budget == 3 + 3 + 6 and not widths.is_uniform()


def test_tree_truncation():
    tree = DraftTree.uniform(3, 4)
    for m in range(1, 5):
        t = tree.truncate(m)
        assert t.depth == m
        assert t.budget == sum(3**l for l in range(1, m + 1))
        assert t.truncate(m) is t  # cached
    assert tree.truncate(9) is tree


# -------------------------------------------------------------------- accounting
def test_standard_sampler_costs_one_nfe_per_step():
    target = LinearGaussianTarget()
    sampler = standard_sampler(target, ConstantSchedule(0.1), num_steps=20)
    result = sampler.sample(np.zeros(4))
    assert result.target_calls == 20
    assert math.isclose(result.speedup, 1.0)
    assert result.trajectory.shape == (21, 4)


def test_full_acceptance_commits_l_steps_per_call():
    """With a perfect proposal and a rule that always accepts, a round commits
    L states, so N steps cost ceil(N / L) target calls."""
    a, N, L, K = 0.9, 24, 4, 3
    target = LinearGaussianTarget(a)
    sampler = SpeculativeSampler(
        target=target,
        proposal=ExactProposal(a),
        schedule=ConstantSchedule(0.1),
        tree=DraftTree.uniform(K, L),
        verifier=AcceptFirstVerifier(),
        num_steps=N,
        check_contract=True,
    )
    result = sampler.sample(np.zeros(4), rng=np.random.default_rng(0))
    assert result.target_calls == N // L
    assert math.isclose(result.speedup, L)
    assert result.acceptance_rate == 1.0
    # one batched call per round over the internal nodes only
    per_round = DraftTree.uniform(K, L).budget // K
    assert result.target_states_evaluated == per_round * (N // L)


def test_truncation_at_the_horizon():
    """N is not a multiple of L: the last round must shrink, not overrun."""
    a, N, L = 0.9, 10, 4
    sampler = SpeculativeSampler(
        target=LinearGaussianTarget(a),
        proposal=ExactProposal(a),
        schedule=ConstantSchedule(0.1),
        tree=DraftTree.uniform(2, L),
        verifier=AcceptFirstVerifier(),
        num_steps=N,
    )
    result = sampler.sample(np.zeros(3), rng=np.random.default_rng(1))
    assert sum(r.committed for r in result.rounds) == N
    assert [r.lookahead for r in result.rounds] == [4, 4, 2]
    assert result.rounds[-1].drafted == DraftTree.uniform(2, 2).budget


def test_delayed_drift_prefetching_costs_one_warmup_call():
    N, L = 12, 3
    target = LinearGaussianTarget()
    sampler = SpeculativeSampler(
        target=target,
        proposal=DelayedDriftProposal(target, prefetch=True),
        schedule=ConstantSchedule(0.05),
        tree=DraftTree.uniform(2, L),
        verifier=ResampleVerifier(),
        num_steps=N,
    )
    result = sampler.sample(np.zeros(5), rng=np.random.default_rng(2))
    # ResampleVerifier never accepts, so every round commits exactly one state:
    # N rounds, plus the single unavoidable warm-up evaluation at n = 0.
    assert result.target_calls == N + 1


# --------------------------------------------------------------------- exactness
def test_trajectory_law_matches_the_reference_sampler():
    """Speculation must not change the law. Compare terminal moments against
    the closed form of the linear-Gaussian chain."""
    a, sigma, N = 0.85, 0.3, 12
    rng = np.random.default_rng(3)
    sampler = SpeculativeSampler(
        target=LinearGaussianTarget(a),
        proposal=ExactProposal(a),
        schedule=ConstantSchedule(sigma),
        tree=DraftTree.uniform(2, 3),
        verifier=AcceptFirstVerifier(),
        num_steps=N,
    )
    finals = np.stack([sampler.sample(np.ones(1), rng=rng).sample for _ in range(4000)])
    mean = a**N
    var = sigma**2 * sum(a ** (2 * k) for k in range(N))
    assert abs(finals.mean() - mean) < 4 * math.sqrt(var / len(finals))
    assert abs(finals.var() / var - 1.0) < 0.08


def test_check_exactness_accepts_valid_and_rejects_invalid_rules():
    ok = check_exactness(ResampleVerifier(), delta=1.2, num_children=3, num_samples=3000, seed=0)
    assert ok.passed, ok
    assert ok.acceptance_rate == 0.0
    bad = check_exactness(BrokenVerifier(), delta=1.2, num_children=3, num_samples=3000, seed=0)
    assert not bad.passed, bad


def test_check_exactness_works_at_delta_zero():
    """delta = 0 must not be a blind spot.

    It is the regime a good proposal approaches, and Remark 2 says every rule
    has to handle it. If the harness recovers the projection direction from the
    request it gets the zero vector there, every projection collapses to 0, and
    the KS statistic against N(0, 1) is exactly 0.5 for *any* rule -- so a
    provably exact rule fails, deterministically, on the one case most worth
    testing.
    """
    for K in (1, 2, 8):
        r = check_exactness(ResampleVerifier(), delta=0.0, num_children=K, seed=0)
        assert r.passed, r
        assert r.ks_statistic < 0.1, f"degenerate projection collapsed: {r}"
    # BrokenVerifier returns the child uncorrected, which at delta = 0 is an
    # exact draw -- the kernels coincide, so it is genuinely correct there
    # (Remark 2). It has to be caught as soon as the means separate.
    assert check_exactness(BrokenVerifier(), delta=0.0, num_children=3, seed=0).passed
    assert not check_exactness(BrokenVerifier(), delta=0.5, num_children=3, seed=0).passed


def test_check_exactness_sweeps_the_regimes_that_differ():
    """The sweep the docs recommend, on a rule known to be exact."""
    for delta in (0.0, 0.1, 1.0, 3.0):
        for K in (1, 2, 8):
            r = check_exactness(
                ResampleVerifier(), delta=delta, num_children=K, seed=0, alpha=0.001
            )
            assert r.passed, r


def test_check_exactness_is_reproducible_from_its_seed():
    """A level-alpha test fails on a correct rule ~alpha of the time. If that
    failure is not reproducible you cannot tell it from a real coupling bug."""
    kw = dict(delta=1.2, num_children=3, num_samples=1500)
    a = check_exactness(ResampleVerifier(), seed=7, **kw)
    b = check_exactness(ResampleVerifier(), seed=7, **kw)
    assert a.ks_statistic == b.ks_statistic
    assert a.acceptance_rate == b.acceptance_rate
    # a different seed really does move the statistic
    c = check_exactness(ResampleVerifier(), seed=8, **kw)
    assert c.ks_statistic != a.ks_statistic


def test_check_exactness_false_failure_rate_is_near_alpha():
    """The seeded harness lets us characterise the test itself: a correct rule
    should clear it for the large majority of seeds."""
    fails = sum(
        not check_exactness(
            ResampleVerifier(), delta=1.0, num_children=2, num_samples=1500, seed=s, alpha=0.01
        ).passed
        for s in range(60)
    )
    assert fails <= 6, f"{fails}/60 false failures; expected only a few at alpha=0.01"


def test_sampler_rejects_a_non_floating_init():
    """An integer state truncates every Gaussian draw to zero and the run would
    complete, report a speedup, and return a silently wrong trajectory."""
    sampler = standard_sampler(LinearGaussianTarget(), ConstantSchedule(0.5), num_steps=6)
    try:
        sampler.sample(np.ones(4, dtype=np.int64))
    except TypeError as exc:
        assert "non-floating" in str(exc)
    else:
        raise AssertionError("an integer init was accepted")
    # float32 remains supported
    assert sampler.sample(np.ones(4, dtype=np.float32)).trajectory.dtype == np.float32


def test_contract_checker_catches_a_lying_accept():
    class Liar(Verifier):
        name = "liar"

        def verify(self, request):
            return VerifyResult(state=request.target_mean, accepted=True, child_index=0)

    sampler = SpeculativeSampler(
        target=LinearGaussianTarget(),
        proposal=ExactProposal(0.9),
        schedule=ConstantSchedule(0.1),
        tree=DraftTree.uniform(2, 2),
        verifier=Liar(),
        num_steps=4,
        check_contract=True,
    )
    try:
        sampler.sample(np.zeros(2))
    except ValueError as exc:
        assert "not child" in str(exc)
    else:
        raise AssertionError("CheckedVerifier failed to catch a mismatched accept")


def test_topology_constraint_is_enforced_at_construction():
    class SingleOnly(Verifier):
        name = "single"
        max_children = 1

        def verify(self, request):  # pragma: no cover
            raise NotImplementedError

    try:
        SpeculativeSampler(
            target=LinearGaussianTarget(),
            proposal=ExactProposal(0.9),
            schedule=ConstantSchedule(0.1),
            tree=DraftTree.uniform(3, 2),
            verifier=SingleOnly(),
            num_steps=4,
        )
    except ValueError as exc:
        assert "at most K=1" in str(exc)
    else:
        raise AssertionError("expected a topology error")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
