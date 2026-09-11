"""End-to-end example: the Gaussian-mixture setting of Section 5.1, in NumPy.

Two things this is meant to show.

1.  What a real model has to supply: one :class:`TargetTransition` and a
    schedule. Everything else is library code.
2.  That the framework is useful *before* any coupling exists. The rule below
    measures the normalised mean mismatch ``delta`` at every node -- the single
    quantity that determines every acceptance probability in the paper -- and
    resamples, so it is exact and costs one step per call. Run it once and you
    know what acceptance rate, and therefore what speedup ceiling, a coupling
    could possibly deliver on your model.

Run: ``python examples/gaussian_mixture.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdiff import (  # noqa: E402
    BatchedSpeculativeSampler,
    DelayedDriftProposal,
    DraftTree,
    NoiseSchedule,
    SpeculativeSampler,
    TargetTransition,
    VerifyRequest,
    VerifyResult,
    Verifier,
    create_verifier,
    standard_sampler,
)
from specdiff.ops import standard_normal_sf  # noqa: E402
from specdiff.verifiers.rank1 import Rank1Frame  # noqa: E402


# ------------------------------------------------------------------- the model
class MixtureReverseKernel(TargetTransition):
    """Euler-Maruyama reverse kernel for a Gaussian-mixture data distribution.

    Under the linear probability path ``x_tau = (1 - tau) x_0 + tau xi`` the
    marginals of a mixture stay a mixture, so the velocity field is available in
    closed form and no network is involved. Coefficients follow eq. (32) and the
    reverse drift eq. (35); the discretisation is eqs. (4)-(5).
    """

    def __init__(self, means, scales, times, *, churn=1.0):
        super().__init__()
        self.mu = np.asarray(means, dtype=np.float64)       # (batch, d)
        self.s = np.asarray(scales, dtype=np.float64)       # (batch,)
        self.times = np.asarray(times, dtype=np.float64)    # (N + 1,) reverse-time grid
        self.gamma = float(times[1] - times[0])
        self.eps = float(churn)
        self.dim = self.mu.shape[1]

    def velocity(self, x, tau):
        """v_tau(x) = E[xi | x] - E[x_0 | x] for the mixture."""
        m = (1.0 - tau) * self.mu                                   # (batch, d)
        v = (1.0 - tau) ** 2 * self.s**2 + tau**2                   # (batch,)
        diff = x[:, None, :] - m[None, :, :]                        # (B, batch, d)
        logw = -0.5 * (diff**2).sum(-1) / v - 0.5 * self.dim * np.log(v)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)                           # posterior weights
        coef = (tau - (1.0 - tau) * self.s**2) / v                  # (batch,)
        per = coef[None, :, None] * diff - self.mu[None, :, :]
        return (w[:, :, None] * per).sum(axis=1)

    def means(self, indices_in_batch, states, steps):
        out = np.empty_like(states)
        for step in sorted(set(steps)):
            idx = [i for i, s in enumerate(steps) if s == step]
            t = self.times[step]
            x = states[idx]
            drift = -(1.0 + self.eps**2) * self.velocity(x, 1.0 - t) - self.eps**2 * x / t
            out[idx] = x + self.gamma * drift
        return out

    def affine(self, step):
        """``(a, b)`` with ``mean = a x + b v``: :meth:`means` written out."""
        t = self.times[step]
        return 1.0 - self.gamma * self.eps**2 / t, -self.gamma * (1.0 + self.eps**2)

    def freeze_drift(self, states, means, steps):
        """The velocity behind ``means``, ``v = (m - a x) / b``, so the delayed
        drift proposal re-applies the churn term at the drafted node's own
        state and step instead of sliding the stale increment along."""
        out = np.empty_like(states)
        for step in sorted(set(steps)):
            idx = [i for i, s in enumerate(steps) if s == step]
            a, b = self.affine(step)
            out[idx] = (means[idx] - a * states[idx]) / b
        return out

    def apply_drift(self, drift, states, steps):
        out = np.empty_like(states)
        for step in sorted(set(steps)):
            idx = [i for i, s in enumerate(steps) if s == step]
            a, b = self.affine(step)
            out[idx] = a * states[idx] + b * drift[idx]
        return out


class MixtureSchedule(NoiseSchedule):
    """sigma_n = eps * sqrt(2 gamma (1 - t_n) / t_n), eq. (37)."""

    def __init__(self, times, churn):
        self.times = np.asarray(times, dtype=np.float64)
        self.gamma = float(times[1] - times[0])
        self.eps = float(churn)

    def sigma(self, step):
        t = self.times[step]
        return self.eps * float(np.sqrt(2.0 * self.gamma * (1.0 - t) / t))


# -------------------------------------------------- a custom rule in 12 lines
class DeltaProbe(Verifier):
    """Records the mean mismatch at every node, then resamples exactly.

    Exact by construction (it ignores the drafts), so it is safe to run on a
    production model, and it answers the only question worth asking before
    committing to a coupling: how far apart are the draft and target kernels?
    """

    name = "delta-probe"

    def __init__(self):
        self.deltas = []

    def reset(self):
        self.deltas.clear()

    def verify(self, request: VerifyRequest) -> VerifyResult:
        frame = Rank1Frame.from_request(request)
        self.deltas.append(frame.delta)
        # Draw from request.rng, not a private stream, or the run stops being
        # reproducible from the seed the caller passed to sample().
        ops = self.backend_for(request)
        noise = ops.randn_stack(1, request.target_mean, request.rng)[0]
        return VerifyResult(request.target_mean + request.sigma * noise, accepted=False)


def plan_batch(alpha: float, lookahead: int, num_steps: int, batch_size: int, trials=200):
    """Cost model for a batch, given a per-step acceptance probability.

    Pure arithmetic -- no sampler involved. Each round a trajectory accepts a
    geometric run capped at L and commits one more state, so it advances
    ``min(Geom(alpha), L) + 1`` steps per target call. Batched cost is the
    *max* over trajectories, because one call serves them all.
    """
    rng = np.random.default_rng(0)
    batched, isolated = [], []
    for _ in range(trials):
        rounds = np.zeros(batch_size, dtype=int)
        steps = np.zeros(batch_size, dtype=int)
        while (steps < num_steps).any():
            live = steps < num_steps
            run = rng.geometric(1.0 - alpha, size=batch_size) - 1  # accepted prefix
            advance = np.minimum(np.minimum(run, lookahead - 1) + 1, num_steps - steps)
            steps = np.where(live, steps + advance, steps)
            rounds += live
        batched.append(num_steps / rounds.max())
        isolated.append(np.mean(num_steps / rounds))
    return float(np.mean(batched)), float(np.mean(isolated))


# ------------------------------------------------------------------------ main
def main() -> None:
    rng = np.random.default_rng(0)
    dim, num_components, N, churn = 8, 5, 50, 1.0
    times = np.linspace(0.05, 0.98, N + 1)

    target = MixtureReverseKernel(
        means=rng.uniform(-2, 2, size=(num_components, dim)),
        scales=rng.uniform(0.10, 0.25, size=num_components),
        times=times,
        churn=churn,
    )
    schedule = MixtureSchedule(times, churn)

    reference = standard_sampler(target, schedule, num_steps=N)
    ref = reference.sample(rng.standard_normal(dim), rng=rng)
    print("reference:", ref.summary())

    tree = DraftTree.uniform(branching=4, lookahead=3)
    probe = DeltaProbe()
    sampler = SpeculativeSampler(
        target=target,
        proposal=DelayedDriftProposal(target),
        schedule=schedule,
        tree=tree,
        verifier=probe,
        num_steps=N,
        check_contract=True,
    )

    deltas = []
    for _ in range(20):
        sampler.sample(rng.standard_normal(dim), rng=rng)
        deltas.extend(probe.deltas)

    delta = float(np.mean(deltas))
    # Per-step acceptance of a single-proposal maximal coupling, eq. (16).
    alpha = 2.0 * standard_normal_sf(delta / 2.0)
    # Chain ceiling from Appendix D.1. A round accepts a Geom(alpha) prefix of
    # mean alpha / (1 - alpha) and then commits one more state on the
    # rejection, so states per target call -> alpha / (1 - alpha) + 1, i.e.
    # 1 / (1 - alpha). The +1 is the point: alpha = 0 must give 1.00x, not 0.
    print(f"\n{tree}")
    print(f"  proposal budget B          {tree.budget}")
    print(f"  verification batch |I|     {len(tree.internal_nodes)}")
    print(f"  mean mismatch delta        {delta:.3f}  (median {np.median(deltas):.3f})")
    print(f"  implied K=1 acceptance     {alpha:.3f}   [eq. 16]")
    print(f"  implied chain ceiling      {1.0 / (1.0 - alpha):.2f}x  [Appendix D.1]")
    # ---- the same thing, batched over images -----------------------------
    batch = 16
    batched = BatchedSpeculativeSampler(
        target=target,
        proposal=DelayedDriftProposal(target),
        schedule=schedule,
        tree=tree,
        verifier=DeltaProbe(),
        num_steps=N,
        keep_trajectories=False,
    )
    br = batched.sample(rng.standard_normal((batch, dim)), rng=rng)
    print(f"\nbatched (batch={batch}): {br.summary()}")

    lookahead = tree.depth
    b_speedup, i_speedup = plan_batch(alpha, lookahead, N, batch)
    print(f"  projected batched speedup  {b_speedup:.2f}x  (batch={batch}, L={lookahead})")
    print(f"  projected isolated speedup {i_speedup:.2f}x")
    print(f"  straggler cost             {100 * (1 - b_speedup / i_speedup):.1f}%")

    print("\nCaveat: a probe that never accepts advances one step per round, so the")
    print("delayed drift is never more than one step stale and this delta is a")
    print("lower bound. Under a real coupling the drift ages across the accepted")
    print("prefix and the mismatch grows -- as it does with dimension, ~sqrt(d).")

    # ---- the projection above, measured -----------------------------------
    # Both of the paper's rules, on the topology each is for: RMC needs a chain
    # (max_children = 1), D-GRS is what the extra width is for.
    print(f"\nMeasured, {lookahead}-level lookahead, mean of 20 trajectories:")
    print(f"{'rule':>7}{'topology':>18}{'B':>5}{'speedup':>10}{'acceptance':>12}")
    for name, rule_tree in (
        ("rmc", DraftTree.chain(lookahead)),
        ("d-grs", DraftTree.chain(lookahead)),
        ("d-grs", tree),
        ("paws", tree),
    ):
        sampler = SpeculativeSampler(
            target=target,
            proposal=DelayedDriftProposal(target),
            schedule=schedule,
            tree=rule_tree,
            verifier=create_verifier(name),
            num_steps=N,
        )
        runs = [sampler.sample(rng.standard_normal(dim), rng=rng) for _ in range(20)]
        shape = f"K={rule_tree.branching}, L={rule_tree.depth}"
        print(
            f"{name:>7}{shape:>18}{rule_tree.budget:>5}"
            f"{np.mean([r.speedup for r in runs]):>10.2f}x"
            f"{np.mean([r.acceptance_rate for r in runs]):>12.3f}"
        )

    print("\nThe two rules agree at K = 1 -- eq. (16) is the ceiling for any")
    print("single-proposal coupling, and RMC attains it. Only D-GRS can spend a")
    print("larger budget on width, which is the point of Algorithm 2. Measured")
    print("acceptance sits below the probe's projection for the reason above.")


if __name__ == "__main__":
    main()
