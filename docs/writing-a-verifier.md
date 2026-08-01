# Writing a verification rule

`Verify` is the one component Algorithm 3 leaves abstract. It is where RMC and D-GRS differ,
where your own coupling goes, and the only place in the library where a bug is silent rather
than loud. This is the guide to writing one.

## The interface

```python
from specdiff import Verifier, VerifyRequest, VerifyResult

class MyRule(Verifier):
    name = "my-rule"
    max_children = None            # or 1 for a single-proposal coupling

    def verify(self, request: VerifyRequest) -> VerifyResult:
        ...
```

That is the whole interface. A rule written this way also runs under the batched sampler
unchanged.

You are given one parent node and its drafted children, and you return one state:

| you get | |
| --- | --- |
| `request.proposal_mean` | `m^p(Y_u)` — the mean the children were drawn around |
| `request.target_mean` | `m^q(Y_u)` — the mean they *should* have been drawn around |
| `request.sigma` | the scale both kernels share at this step |
| `request.children` | `(K, *state_shape)`, **in drafting order** |
| `request.rng` | the run's generator — use this one |
| `request.step`, `.slot`, `.parent_state`, `.info` | context, if you need it |

| you return | |
| --- | --- |
| `state` | an **exact** sample from `N(target_mean, sigma^2 I)` |
| `accepted` | whether `state` is one of the children |
| `child_index` | which one — required iff `accepted` |
| `proposals_examined` | telemetry, optional |

## The three obligations

**1. `state` is an exact sample from `N(target_mean, sigma^2 I)`.**

This is the reason the whole approach is worth anything. It is also the one obligation the
library cannot check per call — see [testing for exactness](#testing-for-exactness).

**2. If `accepted`, `state` *is* `children[child_index]`.**

Not a copy, not a corrected version. The sampler descends into that child's subtree, so a
returned state that merely resembles the child sends the trajectory down the wrong branch and
silently corrupts the chain. `check_contract=True` enforces this.

**3. Children are examined in the order given, if your rule is a sequence coupling.**

`request.children` is in sampling order and the sampler guarantees it is stable. Sorting them,
or examining them by likelihood ratio, breaks exactness for a sequence coupling like
Algorithm 2.

## Rank-1 coordinates

Eq. (24) guarantees `P` and `Q` are isotropic Gaussians differing only in mean. So everything
orthogonal to `mu_q - mu_p` has the same law under both, and the `d`-dimensional coupling
collapses to a scalar one:

```
S ~ N(0, 1)      under P
S ~ N(delta, 1)  under Q        delta = ||mu_q - mu_p|| / sigma
```

A rule only has to move the scalar `S`. The orthogonal residual of whichever proposal it keeps
is carried through untouched. `Rank1Frame` is the change of coordinates:

```python
from specdiff.verifiers.rank1 import Rank1Frame

frame = Rank1Frame.from_request(request)
frame.delta                       # normalised mean mismatch — sets every acceptance prob
frame.direction                   # the unit vector e
s, z_perp = frame.project(y)      # eq. (10):  Y -> (S, Z_perp)
y = frame.reconstruct(s, z_perp)  # eq. (11):  (S, Z_perp) -> Y
```

`delta` is the single number that controls every acceptance probability in the paper — eq. (16)
for RMC, eq. (15) for D-GRS. It is worth measuring on your own model before committing to a
coupling; see [the delta probe](#measuring-your-headroom-first).

This lives in the library rather than in any one rule because getting the reconstruction wrong
is a silent-correctness bug: the sample still looks Gaussian, just not from the right
distribution.

### Degeneracy

Always check `frame.degenerate` before dividing by `delta`:

```python
frame = Rank1Frame.from_request(request)
if frame.degenerate:              # the two kernels coincide, so accept anything
    return VerifyResult(request.child(0), accepted=True, child_index=0)
tau = frame.tau(lam)              # ln(lam) / delta, guarded
```

`degenerate` is `delta <= tol`, **not** `delta == 0`, with `tol = sqrt(eps)` of the state
dtype — about `1.5e-8` in float64 and `3.4e-4` in float32.

Exact equality is the wrong predicate, and the reason is worth internalising: the regime that
breaks a rule is small-and-nonzero `delta`, which is precisely what a *good* proposal produces.
At `delta = 1e-16` an `== 0` test says "not degenerate", `tau = ln(lambda) / delta` comes out
around `-7e15`, `Phi_bar` saturates to exactly `0`, and the D-GRS residual mass `G_k` becomes
zero — a division by zero one step later, from a proposal that was doing its job well. Below
`tol` the two kernels are indistinguishable at the state's own precision (their TV distance is
`~delta / sqrt(2 pi)`), so accepting unconditionally is the correct limit, not an
approximation.

`frame.tau(level)` raises `ZeroDivisionError` on a degenerate frame rather than handing back an
infinity that would propagate quietly.

## Randomness

Draw from `request.rng`, never a private generator:

```python
ops = self.backend_for(request)                       # backend without importing ops
noise = ops.randn_stack(1, request.target_mean, request.rng)[0]
u = ops.uniform(request.rng)                          # scalar in [0, 1)
```

A rule that reaches for `np.random` directly breaks reproducibility from the seed the caller
passed to `sample()`, and makes a failed exactness test impossible to reproduce. `backend_for`
is a convenience on `Verifier` so subclasses need no framework import at all.

## Worked example: a diagnostic rule

The simplest useful rule. It is exact by construction because it ignores the drafts entirely,
which means it is safe to run against a production model, and it answers the question worth
asking before you commit to a coupling.

```python
from specdiff import Verifier, VerifyRequest, VerifyResult
from specdiff.verifiers.rank1 import Rank1Frame

class DeltaProbe(Verifier):
    """Records the mean mismatch at every node, then resamples exactly."""

    name = "delta-probe"

    def __init__(self):
        self.deltas = []

    def reset(self):                      # called at the start of each sample()
        self.deltas.clear()

    def verify(self, request: VerifyRequest) -> VerifyResult:
        frame = Rank1Frame.from_request(request)
        self.deltas.append(frame.delta)

        ops = self.backend_for(request)
        noise = ops.randn_stack(1, request.target_mean, request.rng)[0]
        return VerifyResult(
            state=request.target_mean + request.sigma * noise,
            accepted=False,
        )
```

Three things to copy from it: `reset()` drops per-run state, the noise comes from
`request.rng`, and a rejection returns `accepted=False` with **no** `child_index` —
`VerifyResult` rejects the inconsistent combinations in `__post_init__`.

## Measuring your headroom first

Run the probe on your model and convert the observed `delta` into the best speedup any
single-proposal coupling could reach:

```python
import numpy as np
from specdiff import DraftTree, DelayedDriftProposal, SpeculativeSampler
from specdiff.ops import standard_normal_sf

probe = DeltaProbe()
sampler = SpeculativeSampler(
    target=my_target, proposal=DelayedDriftProposal(my_target),
    schedule=my_schedule, tree=DraftTree.uniform(4, 3),
    verifier=probe, num_steps=100, check_contract=True,
)
for _ in range(20):
    sampler.sample(rng.standard_normal(dim), rng=rng)

delta = float(np.mean(probe.deltas))
alpha = 2.0 * standard_normal_sf(delta / 2.0)   # per-step acceptance, eq. (16)
print(f"delta {delta:.3f}  acceptance {alpha:.3f}  chain ceiling {1/(1-alpha):.2f}x")
```

The ceiling is `1 / (1 - alpha)`: a round accepts a `Geom(alpha)` prefix of mean
`alpha / (1 - alpha)` and then commits one more state on the rejection.

One caveat, and it matters: a probe that never accepts advances one step per round, so a
delayed drift is never more than one step stale and the measured `delta` is a **lower bound**.
Under a real coupling the drift ages across the accepted prefix and the mismatch grows — as it
also does with dimension, roughly like `sqrt(d)`.

`examples/gaussian_mixture.py` runs this end to end.

## Testing for exactness

Obligation 1 cannot be checked per call, so test it:

```python
from specdiff import check_exactness

report = check_exactness(MyRule(), delta=1.5, num_children=4, seed=0)
assert report.passed, report
print(report)
# [PASS] delta=1.500 K=4 n=4000 KS=0.0119 (crit 0.0257) accept=0.612
```

The test projects the returned state onto the displacement direction, where exactness implies
`N(delta, 1)` **regardless of what the rule did internally**, and runs a one-sample KS test on
that scalar. It catches the coupling bugs that produce plausible-looking but wrong samples,
and needs no SciPy.

Sweep the regimes that actually differ, and **include `delta = 0`** — it is the case a good
proposal approaches, the one Remark 2 forces every rule to special-case, and therefore the one
most likely to be wrong:

```python
for delta in (0.0, 0.1, 1.0, 3.0):
    for K in (1, 2, 8):
        r = check_exactness(MyRule(), delta=delta, num_children=K, seed=0, alpha=0.001)
        assert r.passed, r
```

Note the tighter `alpha`. A sweep runs many tests, so at the default `alpha=0.01` the chance
that *some* cell trips on a correct rule grows with the number of cells; `0.001` buys back the
headroom at a small cost in sensitivity. Use the default for a single check.

**Keep the seed.** This is a hypothesis test at level `alpha`, so a *correct* rule fails it
about `alpha` of the time. `seed` controls every draw — the direction, the children, and
whatever your rule consumes through `request.rng` — so a report is reproducible and a failure
is something you can actually debug. Without it, an occasional red build is indistinguishable
from a real coupling bug.

`report.acceptance_rate` and `report.mean_examined` are the other half of the picture: a rule
can be perfectly exact and still useless if it never accepts.

## While developing: `check_contract=True`

```python
sampler = SpeculativeSampler(..., verifier=MyRule(), check_contract=True)
```

Wraps your rule in `CheckedVerifier`, which costs a couple of comparisons per node and catches
the checkable half of the contract: wrong output type, wrong state shape, non-finite values,
an out-of-range `child_index`, and — the important one — `accepted=True` with a state that is
not the drafted child.

The checks are identical on both drivers, so a rule the scalar sampler rejects is rejected
under batching too, with the same message plus a row number.

## Restricting the topology

If your rule can only couple one proposal, say so:

```python
class MyMaximalCoupling(Verifier):
    max_children = 1
```

`check_topology` then refuses a branching tree **at construction time** rather than silently
ignoring siblings at every node:

```
ValueError: MyMaximalCoupling supports at most K=1 proposals per node, but the
draft tree has K=3. Use DraftTree.chain(L) for single-proposal rules.
```

## Vectorising for the batched sampler

Rules get `verify_batch` for free — the default loops over rows calling your `verify`, so
nothing needs rewriting. Override it when the per-node work is worth vectorising: for the
paper's rules, the scalar sweep over levels `lambda_k` becomes a `(batch,)` vector operation
while the `d`-dimensional projections and reconstructions become single batched ops.

```python
def verify_batch(self, request: BatchedVerifyRequest) -> BatchedVerifyResult:
    ...
    return BatchedVerifyResult(states=..., accepted=(...), child_index=(...))
```

Two rules for an override:

- **Stay row-independent.** Row `j` may depend only on `request.row(j)`. Coupling rows to each
  other changes the joint law of the batch even if each marginal still looks right.
- **`sigmas` is per row**, not scalar — rows belong to different steps. `ops.scale_rows` is the
  portable way to broadcast it.

Prove the override is an optimisation and not a behaviour change by running both paths and
comparing; `tests/test_batched.py::test_vectorised_verify_batch_agrees_with_the_row_loop` shows
the pattern.

## Registering a rule by name

```python
from specdiff import register_verifier, create_verifier

@register_verifier("my-rule")
class MyRule(Verifier):
    ...

rule = create_verifier("my-rule")          # addressable from a config file
```

## Implementing the paper's two algorithms

`specdiff/verifiers/stubs.py` holds `ReflectionMaximalCoupling` (Algorithm 1) and
`GreedyRejectionSampling` (Algorithm 2). They are deliberately unimplemented — their purpose
is to fix the names, topology constraints and telemetry so that filling them in is a local
edit, and the bodies are left as the reader's work. What follows is what you need, not the
answer.

**Algorithm 1 — reflection maximal coupling, `K = 1`.** Project the single child to `s_hat`;
accept with probability `1 ^ phi(s_hat - delta) / phi(s_hat)`; on rejection reflect,
`s = delta - s_hat`; reconstruct. Per-step acceptance is `2 * Phi_bar(-delta / 2)`, eq. (16).
Report `proposals_examined = 1`.

**Algorithm 2 — greedy rejection sampling, any `K`.** Sweep the children *in drafting order* —
this is the sequence coupling, not the list coupling. Maintain the level `lambda_k` and the
residual mass `G_{k+1}`, and accept child `k` with probability
`1 ^ (rho(s_k) - lambda_{k-1})_+ / G_k`. On a full sweep of rejections, sample the normalised
residual (eq. 13). The masses of the super-level sets are half-space masses in the projected
coordinate — `Q(H_k) = Phi_bar(tau_k - delta/2)` and `P(H_k) = Phi_bar(tau_k + delta/2)` with
`tau_k = ln(lambda_k) / delta` (Appendix B.2) — so no numerical integration is needed. Report
`proposals_examined = k` on acceptance and `K + 1` on the residual branch.

`specdiff.ops` provides `standard_normal_cdf` and `standard_normal_sf` so neither needs SciPy,
and `Rank1Frame.tau` computes `tau_k` with the degeneracy guard already applied.

### Three traps

1. **Order.** The children arrive in sampling order and a sequence coupling must keep it.
   Sorting them or examining them by likelihood ratio breaks exactness.
2. **Which orthogonal residual you carry through.** Algorithm 2 returns `Z_perp,k` on
   acceptance of child `k` but `Z_perp,1` on the residual branch (lines 9 and 18). Using the
   wrong one produces samples that pass a casual eyeball check and fail `check_exactness`.
3. **Degeneracy.** Check `frame.degenerate` before computing any `tau`. See
   [above](#degeneracy).

## Checklist

- [ ] `verify` returns an exact draw from `N(target_mean, sigma^2 I)`
- [ ] accepted ⟹ the returned state *is* `children[child_index]`
- [ ] children examined in the given order, if sequence-coupled
- [ ] `frame.degenerate` checked before any division by `delta`
- [ ] all randomness drawn from `request.rng`
- [ ] `reset()` clears per-run state
- [ ] `max_children` set if the rule is single-proposal
- [ ] `check_exactness` passes across a sweep of `delta` and `K`, with a fixed seed
- [ ] a run with `check_contract=True` completes
- [ ] any `verify_batch` override agrees with the row-wise loop
