"""Tests for the batched sampler. `python tests/test_batched.py`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdiff import (  # noqa: E402
    BatchedDelayedDriftProposal,
    BatchedSpeculativeSampler,
    BatchedVerifyRequest,
    BatchedVerifyResult,
    ConstantSchedule,
    DelayedDriftProposal,
    DraftTree,
    PerSlotProposal,
    ResampleVerifier,
    SpeculativeSampler,
    StatelessBatchedProposal,
    Verifier,
    VerifyResult,
)
from specdiff.ops import resolve_backend  # noqa: E402

from test_sampler import (  # noqa: E402
    AcceptFirstVerifier,
    ExactProposal,
    LinearGaussianTarget,
)

A, SIGMA = 0.9, 0.2


class CoinFlipVerifier(Verifier):
    """Accepts a uniformly chosen child with probability ``p``, else resamples.

    Exact only when the proposal equals the target (delta = 0), which is the
    regime the sampler tests run in. Its point is that acceptance varies row to
    row, so trajectories in a batch genuinely desynchronise.
    """

    name = "coin"

    def __init__(self, p=0.5, seed=0):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def verify(self, request):
        if self.rng.random() < self.p:
            k = int(self.rng.integers(request.num_children))
            return VerifyResult(request.child(k), accepted=True, child_index=k)
        noise = self.rng.standard_normal(request.state_shape)
        return VerifyResult(request.target_mean + request.sigma * noise, accepted=False)


class ParityVerifier(Verifier):
    """A deterministic rule, so the two code paths can be compared exactly.

    Not exact -- it is a plumbing fixture, not a coupling. It accepts a child
    chosen by the step index unless ``(step + slot) % 3 == 0``, which is enough
    to desynchronise the batch in a reproducible way. Overrides
    :meth:`verify_batch` to exercise the vectorised path.
    """

    name = "parity"

    @staticmethod
    def _decide(step, slot, k):
        return None if (step + slot) % 3 == 0 else step % k

    def verify(self, request):
        pick = self._decide(request.step, request.slot, request.num_children)
        if pick is None:
            return VerifyResult(request.target_mean, accepted=False)
        return VerifyResult(request.child(pick), accepted=True, child_index=pick)

    def verify_batch(self, request: BatchedVerifyRequest) -> BatchedVerifyResult:
        ops = resolve_backend(request.children)
        picks = [
            self._decide(s, slot, request.num_children)
            for s, slot in zip(request.steps, request.slots)
        ]
        states = ops.stack_rows(
            [
                request.target_mean[j] if p is None else request.children[j][p]
                for j, p in enumerate(picks)
            ]
        )
        return BatchedVerifyResult(
            states=states,
            accepted=tuple(p is not None for p in picks),
            child_index=tuple(picks),
        )


class LoopOnlyParityVerifier(ParityVerifier):
    """Same rule with the override removed, forcing the row-wise fallback."""

    name = "parity-loop"
    verify_batch = Verifier.verify_batch

    def verify(self, request):
        pick = self._decide(request.step, request.slot, request.num_children)
        if pick is None:
            return VerifyResult(request.target_mean, accepted=False)
        return VerifyResult(request.child(pick), accepted=True, child_index=pick)


def _sampler(verifier, *, batch=8, N=24, K=3, L=4, proposal=None, **kw):
    target = LinearGaussianTarget(A)
    return target, BatchedSpeculativeSampler(
        target=target,
        proposal=proposal or StatelessBatchedProposal(ExactProposal(A)),
        schedule=ConstantSchedule(SIGMA),
        tree=DraftTree.uniform(K, L),
        verifier=verifier,
        num_steps=N,
        **kw,
    )


# ------------------------------------------------------------------- accounting
class DeterministicVerifier(Verifier):
    """Consumes no randomness, so any scalar/batched divergence is the sampler's.

    Not exact -- a plumbing fixture. It rejects every third step, which is
    enough to exercise mid-tree rejection and the truncation path.
    """

    name = "deterministic"

    def verify(self, request):
        if request.step % 3 == 0:
            return VerifyResult(request.target_mean, accepted=False)
        k = request.step % request.num_children
        return VerifyResult(request.child(k), accepted=True, child_index=k)


def test_batch_of_one_matches_the_scalar_sampler():
    N, K, L = 24, 3, 4
    target_b, batched = _sampler(AcceptFirstVerifier(), batch=1, N=N, K=K, L=L)
    rb = batched.sample(np.zeros((1, 4)), rng=np.random.default_rng(0))

    target_s = LinearGaussianTarget(A)
    scalar = SpeculativeSampler(
        target=target_s,
        proposal=ExactProposal(A),
        schedule=ConstantSchedule(SIGMA),
        tree=DraftTree.uniform(K, L),
        verifier=AcceptFirstVerifier(),
        num_steps=N,
    )
    rs = scalar.sample(np.zeros(4), rng=np.random.default_rng(0))

    assert rb.target_calls == rs.target_calls
    assert rb.target_states_evaluated == rs.target_states_evaluated
    assert rb.drafted_states == rs.drafted_states
    assert math.isclose(rb.speedup, rs.speedup)


def test_batch_of_one_reproduces_the_scalar_trajectory_exactly():
    """The counts agreeing is weak; the *samples* must agree too.

    Both samplers reach the same draft nodes by different routes -- the scalar
    one truncates the tree, the batched one filters levels by per-row lookahead
    -- so they must consume the RNG in the same order. Anything that reorders
    the draws in ``_draft`` shows up here and nowhere else.
    """
    N, K, L = 21, 3, 4
    scalar = SpeculativeSampler(
        target=LinearGaussianTarget(A),
        proposal=ExactProposal(A),
        schedule=ConstantSchedule(SIGMA),
        tree=DraftTree.uniform(K, L),
        verifier=DeterministicVerifier(),
        num_steps=N,
    )
    rs = scalar.sample(np.zeros(4), rng=np.random.default_rng(0))
    _, batched = _sampler(
        DeterministicVerifier(), batch=1, N=N, K=K, L=L, keep_trajectories=True
    )
    rb = batched.sample(np.zeros((1, 4)), rng=np.random.default_rng(0))
    assert np.array_equal(rs.trajectory, rb.trajectories[0])


def test_batch_of_one_matches_the_scalar_sampler_with_a_stateful_proposal():
    """Same, with a delayed drift: PerSlotProposal must reproduce it per slot."""
    N, K, L = 21, 3, 4
    t_s, t_b = LinearGaussianTarget(A), LinearGaussianTarget(A)
    scalar = SpeculativeSampler(
        target=t_s,
        proposal=DelayedDriftProposal(t_s),
        schedule=ConstantSchedule(SIGMA),
        tree=DraftTree.uniform(K, L),
        verifier=DeterministicVerifier(),
        num_steps=N,
    )
    rs = scalar.sample(np.zeros(4), rng=np.random.default_rng(0))
    batched = BatchedSpeculativeSampler(
        target=t_b,
        proposal=PerSlotProposal(lambda: DelayedDriftProposal(t_b)),
        schedule=ConstantSchedule(SIGMA),
        tree=DraftTree.uniform(K, L),
        verifier=DeterministicVerifier(),
        num_steps=N,
        keep_trajectories=True,
    )
    rb = batched.sample(np.zeros((1, 4)), rng=np.random.default_rng(0))
    assert rb.target_calls == rs.target_calls
    assert np.array_equal(rs.trajectory, rb.trajectories[0])


def test_one_target_call_serves_the_whole_batch():
    N, L, batch, K = 24, 4, 16, 3
    _, sampler = _sampler(AcceptFirstVerifier(), batch=batch, N=N, K=K, L=L)
    r = sampler.sample(np.zeros((batch, 4)), rng=np.random.default_rng(1))
    assert r.target_calls == N // L  # not batch * N // L
    assert r.samples.shape == (batch, 4)
    assert r.occupancy == 1.0
    assert math.isclose(r.speedup, L)


def test_stragglers_make_the_batch_slower_than_its_members():
    N, batch = 40, 24
    _, sampler = _sampler(CoinFlipVerifier(p=0.6, seed=2), batch=batch, N=N, K=2, L=4)
    r = sampler.sample(np.zeros((batch, 3)), rng=np.random.default_rng(3))
    assert all(n == N for n in _steps_reached(r))
    # a batch advances at the pace of its slowest live member
    assert r.speedup <= r.mean_isolated_speedup + 1e-9
    assert r.speedup > 1.0
    assert 0.0 < r.occupancy <= 1.0
    assert r.target_calls == max(r.rounds_per_trajectory)


def _steps_reached(result):
    reached = [0] * result.batch_size
    for rec in result.rounds:
        for slot, c in zip(rec.active, rec.committed):
            reached[slot] += c
    return reached


def test_trajectories_are_kept_and_consistent():
    N, batch = 12, 5
    _, sampler = _sampler(
        CoinFlipVerifier(p=0.5, seed=4), batch=batch, N=N, K=2, L=3, keep_trajectories=True
    )
    r = sampler.sample(np.ones((batch, 2)), rng=np.random.default_rng(5))
    assert r.trajectories.shape == (batch, N + 1, 2)
    assert np.allclose(r.trajectories[:, -1], r.samples)
    assert np.allclose(r.trajectories[:, 0], 1.0)


def test_horizon_truncation_per_trajectory():
    """Trajectories reach the horizon at different times; each must stop at N."""
    N, batch = 17, 6
    _, sampler = _sampler(CoinFlipVerifier(p=0.7, seed=6), batch=batch, N=N, K=2, L=5)
    r = sampler.sample(np.zeros((batch, 2)), rng=np.random.default_rng(7))
    assert _steps_reached(r) == [N] * batch
    # once a trajectory finishes it leaves the batch
    assert len(r.rounds[-1].active) <= batch
    assert r.occupancy < 1.0


# --------------------------------------------------------------------- exactness
def test_batched_trajectory_law_is_unchanged():
    N, batch = 12, 200
    target = LinearGaussianTarget(A)
    sampler = BatchedSpeculativeSampler(
        target=target,
        proposal=StatelessBatchedProposal(ExactProposal(A)),
        schedule=ConstantSchedule(SIGMA),
        tree=DraftTree.uniform(3, 3),
        verifier=CoinFlipVerifier(p=0.5, seed=8),
        num_steps=N,
        check_contract=True,
    )
    rng = np.random.default_rng(9)
    finals = np.concatenate(
        [sampler.sample(np.ones((batch, 1)), rng=rng).samples.ravel() for _ in range(20)]
    )
    mean = A**N
    var = SIGMA**2 * sum(A ** (2 * k) for k in range(N))
    assert abs(finals.mean() - mean) < 4 * math.sqrt(var / len(finals))
    assert abs(finals.var() / var - 1.0) < 0.06


def test_vectorised_verify_batch_agrees_with_the_row_loop():
    """Overriding verify_batch is an optimisation, never a behaviour change."""
    N, batch = 20, 12
    out = {}
    for name, rule in (("loop", LoopOnlyParityVerifier()), ("vec", ParityVerifier())):
        _, sampler = _sampler(rule, batch=batch, N=N, K=2, L=3, keep_trajectories=True)
        out[name] = sampler.sample(np.zeros((batch, 2)), rng=np.random.default_rng(12))
    assert out["loop"].target_calls == out["vec"].target_calls
    assert out["loop"].rounds_per_trajectory == out["vec"].rounds_per_trajectory
    assert np.allclose(out["loop"].samples, out["vec"].samples)
    assert np.allclose(out["loop"].trajectories, out["vec"].trajectories)


# --------------------------------------------------------------------- proposals
def test_stateful_proposal_cannot_be_shared_across_a_batch():
    target = LinearGaussianTarget(A)
    try:
        StatelessBatchedProposal(DelayedDriftProposal(target))
    except TypeError as exc:
        assert "stateful" in str(exc)
    else:
        raise AssertionError("a stateful proposal was silently shared across the batch")


def test_batched_delayed_drift_warms_up_once_for_the_whole_batch():
    N, batch = 12, 7
    target = LinearGaussianTarget(A)
    sampler = BatchedSpeculativeSampler(
        target=target,
        proposal=BatchedDelayedDriftProposal(target, prefetch=True),
        schedule=ConstantSchedule(0.05),
        tree=DraftTree.uniform(2, 3),
        verifier=ResampleVerifier(),
        num_steps=N,
    )
    r = sampler.sample(np.zeros((batch, 4)), rng=np.random.default_rng(13))
    # ResampleVerifier never accepts: N iterations, plus ONE batched warm-up
    # call covering all batch trajectories (not batch separate ones).
    assert r.target_calls == N + 1
    assert _steps_reached(r) == [N] * batch


def test_per_slot_proposal_keeps_state_separate():
    N, batch = 12, 4
    target = LinearGaussianTarget(A)
    sampler = BatchedSpeculativeSampler(
        target=target,
        proposal=PerSlotProposal(lambda: DelayedDriftProposal(target, prefetch=True)),
        schedule=ConstantSchedule(0.05),
        tree=DraftTree.uniform(2, 2),
        verifier=ResampleVerifier(),
        num_steps=N,
    )
    r = sampler.sample(np.zeros((batch, 3)), rng=np.random.default_rng(14))
    # one warm-up per slot here, since the instances are independent
    assert r.target_calls == N + batch
    assert _steps_reached(r) == [N] * batch


def test_irregular_tree_is_rejected_with_a_useful_message():
    target = LinearGaussianTarget(A)
    pruned = DraftTree([-1, 0, 0, 1, 1, 2])  # node 1 has 2 children, node 2 has 1
    try:
        BatchedSpeculativeSampler(
            target=target,
            proposal=StatelessBatchedProposal(ExactProposal(A)),
            schedule=ConstantSchedule(SIGMA),
            tree=pruned,
            verifier=AcceptFirstVerifier(),
            num_steps=4,
        )
    except ValueError as exc:
        assert "level-uniform" in str(exc)
    else:
        raise AssertionError("expected a level-uniformity error")


# ------------------------------------------------------ sampler-parity guarantees
def test_batched_sampler_rejects_a_non_floating_init():
    """An integer init would truncate every noise draw to zero, silently."""
    _, sampler = _sampler(AcceptFirstVerifier(), batch=3, N=6, K=2, L=2)
    try:
        sampler.sample(np.ones((3, 4), dtype=np.int64))
    except TypeError as exc:
        assert "non-floating" in str(exc)
    else:
        raise AssertionError("an integer init was accepted")


def test_contract_checking_is_as_strict_batched_as_scalar():
    """check_contract=True must not weaken when you move to the batched sampler.

    CheckedVerifier.verify_batch delegates to inner.verify_batch, so
    CheckedVerifier.verify never runs on this path; the per-row checks have to
    be applied there too or a malformed state reaches the buffer and surfaces
    as an opaque backend broadcast error.
    """

    class WrongShape(Verifier):
        name = "wrong-shape"

        def verify(self, request):
            return VerifyResult(np.zeros(request.state_shape[0] + 1), accepted=False)

    _, sampler = _sampler(WrongShape(), batch=3, N=4, K=2, L=2, check_contract=True)
    try:
        sampler.sample(np.zeros((3, 3)))
    except ValueError as exc:
        assert "expected (3,)" in str(exc) and "row" in str(exc)
    else:
        raise AssertionError("a wrong-shaped state was not caught on the batched path")


def test_lying_accept_is_caught_batched_too():
    class Liar(Verifier):
        name = "liar"

        def verify(self, request):
            return VerifyResult(state=request.target_mean, accepted=True, child_index=0)

    _, sampler = _sampler(Liar(), batch=3, N=4, K=2, L=2, check_contract=True)
    try:
        sampler.sample(np.zeros((3, 2)))
    except ValueError as exc:
        assert "not child" in str(exc)
    else:
        raise AssertionError("CheckedVerifier failed to catch a mismatched accept")


def test_info_node_is_available_on_both_samplers():
    """A rule keyed on info['node'] must not KeyError under batching."""
    seen = {"scalar": [], "batched": []}

    class NodeReader(Verifier):
        name = "node-reader"

        def __init__(self, key):
            self.key = key

        def verify(self, request):
            seen[self.key].append(request.info["node"])
            return VerifyResult(request.child(0), accepted=True, child_index=0)

    SpeculativeSampler(
        target=LinearGaussianTarget(A),
        proposal=ExactProposal(A),
        schedule=ConstantSchedule(SIGMA),
        tree=DraftTree.uniform(2, 2),
        verifier=NodeReader("scalar"),
        num_steps=4,
    ).sample(np.zeros(3), rng=np.random.default_rng(0))

    _, sampler = _sampler(NodeReader("batched"), batch=1, N=4, K=2, L=2)
    sampler.sample(np.zeros((1, 3)), rng=np.random.default_rng(0))

    assert seen["scalar"] and seen["scalar"] == seen["batched"]
    # the batch-wide tuple is what a vectorised rule sees
    assert all(isinstance(u, int) for u in seen["batched"])


def test_batched_acceptance_rate_matches_the_rule():
    """The paper's headline diagnostic has to be recoverable from a batched run."""
    N, batch, p = 60, 40, 0.6
    _, sampler = _sampler(CoinFlipVerifier(p=p, seed=1), batch=batch, N=N, K=2, L=3)
    r = sampler.sample(np.zeros((batch, 3)), rng=np.random.default_rng(0))
    assert abs(r.acceptance_rate - p) < 0.05
    # and it agrees with what the scalar sampler reports for the same rule
    scalar = SpeculativeSampler(
        target=LinearGaussianTarget(A),
        proposal=ExactProposal(A),
        schedule=ConstantSchedule(SIGMA),
        tree=DraftTree.uniform(2, 3),
        verifier=CoinFlipVerifier(p=p, seed=1),
        num_steps=N,
    )
    rs = scalar.sample(np.zeros(3), rng=np.random.default_rng(0))
    assert abs(r.acceptance_rate - rs.acceptance_rate) < 0.1
    # accepted_depth is consistent with committed, round by round
    for rec in r.rounds:
        for c, a, rej in zip(rec.committed, rec.accepted_depth, rec.rejected):
            assert a == (c - 1 if rej else c)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
