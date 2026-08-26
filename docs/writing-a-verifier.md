# Writing a verification rule

Algorithm 3 leaves `Verify` abstract. RMC, D-GRS, and custom couplings implement this
component. This guide defines the interface, correctness requirements, and validation process.

## The interface

```python
from specdiff import Verifier, VerifyRequest, VerifyResult

class MyRule(Verifier):
    name = "my-rule"
    max_children = None            # or 1 for a single-proposal coupling

    def verify(self, request: VerifyRequest) -> VerifyResult:
        ...
```

This interface supports both scalar and batched samplers without modification.

You are given one parent node and its drafted children, and you return one state:

| you get | |
| --- | --- |
| `request.proposal_mean` | `m^p(Y_u)` — the mean the children were drawn around |
| `request.target_mean` | `m^q(Y_u)` — the mean they *should* have been drawn around |
| `request.sigma` | the scale both kernels share at this step |
| `request.children` | `(K, *state_shape)`, **in drafting order** |
| `request.rng` | the run's generator — use this one |
| `request.step`, `.index_in_batch`, `.parent_state`, `.info` | context, if you need it |

| you return | |
| --- | --- |
| `state` | an **exact** sample from `N(target_mean, sigma^2 I)` |
| `accepted` | whether `state` is one of the children |
| `child_index` | which one — required iff `accepted` |
| `proposals_examined` | telemetry, optional |

## The three obligations

**1. `state` is an exact sample from `N(target_mean, sigma^2 I)`.**

This requirement preserves the target distribution. The library cannot validate it from one
call; see [testing for exactness](#testing-for-exactness).

**2. If `accepted`, `state` *is* `children[child_index]`.**

The result must be the selected child, not a corrected copy. The sampler descends through that
child's subtree, so returning a different state invalidates the trajectory. Set
`check_contract=True` to enforce this condition.

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

The library centralizes this transformation because an incorrect reconstruction can return a
Gaussian sample from the wrong distribution without raising an error.

### Degeneracy

Always check `frame.degenerate` before dividing by `delta`:

```python
frame = Rank1Frame.from_request(request)
if frame.degenerate:              # the two kernels coincide, so accept anything
    return VerifyResult(request.child(0), accepted=True, child_index=0)
tau = frame.tau(lam)              # ln(lam) / delta, guarded
```

`degenerate` is `delta <= tol`, **not** `delta == 0`, with `tol = 1e-10`
(`DEFAULT_DEGENERATE_TOL`) — a constant, and deliberately *not* derived from the state dtype.
Everything that can break down here — `delta`, `tau = ln(lambda)/delta`, the D-GRS masses
`G_k` — is a Python float computed in float64 whatever the states carry, so the floor is a
property of float64, not of the array. That floor is `delta ~ 1e-15`, where
`G_2 = Phi_bar(-delta/2) - Phi_bar(delta/2)` cancels to zero; `1e-10` clears it by five orders
of magnitude while admitting only `~4e-11` of total variation.

The shortcut accepts unconditionally and introduces `TV = delta / sqrt(2 pi)` whenever it is
used. A loose tolerance therefore reduces exactness in the small-`delta` regime. The previous
`sqrt(eps)` rule reached `3.1e-2` for float16, which `check_exactness` identifies as non-exact.

Do not test degeneracy with exact equality. A small nonzero `delta` can produce
`tau = ln(lambda) / delta` with extreme magnitude, saturate `Phi_bar` to zero, and make the
D-GRS residual mass `G_k` vanish. At `delta = 1e-16`, for example, `tau` is approximately
`-7e15`. Below `tol`, the kernels are indistinguishable at the state's precision, with total
variation approximately `delta / sqrt(2 pi)`, so unconditional acceptance implements the
limiting case.

`frame.tau(level)` raises `ZeroDivisionError` on a degenerate frame rather than handing back an
infinity that would propagate quietly.

## Randomness

Draw all random values from `request.rng`:

```python
ops = self.backend_for(request)                       # backend without importing ops
noise = ops.randn_stack(1, request.target_mean, request.rng)[0]
u = ops.uniform(request.rng)                          # scalar in [0, 1)
```

Using a private generator such as `np.random` breaks reproducibility from the seed passed to
`sample()`. `backend_for` lets verifier subclasses access random operations without importing
a framework directly.

## Worked example: a diagnostic rule

This diagnostic verifier is exact by construction because it ignores drafted states. It can
therefore measure proposal mismatch on a model before a coupling is selected.

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

Because the probe always rejects, it advances one step per round and its delayed drift is at
most one step stale. The measured `delta` is therefore a **lower bound**. With an accepting
coupling, drift can age across the committed prefix and increase the mismatch. Mismatch also
typically grows with dimension at approximately `sqrt(d)`.

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
that scalar. This detects distributional errors in the coupling and requires no SciPy.

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
may be exact while providing no acceleration if it always rejects.

## While developing: `check_contract=True`

```python
sampler = SpeculativeSampler(..., verifier=MyRule(), check_contract=True)
```

Wraps your rule in `CheckedVerifier`, which costs a couple of comparisons per node and catches
the checkable half of the contract: wrong output type, wrong state shape, non-finite values,
an out-of-range `child_index`, and — the important one — `accepted=True` with a state that is
not the drafted child.

The checks are identical on both samplers, so a rule the scalar sampler rejects is rejected
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

The default `verify_batch` implementation calls `verify` for each row, so a separate batched
implementation is optional. Override it when per-node work benefits from vectorization. For
the paper's rules, the scalar sweep over `lambda_k` becomes a `(batch,)` vector operation,
while the `d`-dimensional projections and reconstructions become batched operations.

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

## The paper's two algorithms

`specdiff/verifiers/rmc.py` holds `ReflectionMaximalCoupling` (Algorithm 1) and
`specdiff/verifiers/dgrs.py` holds `GreedyRejectionSampling` (Algorithm 2).

**Algorithm 1 is implemented**, and is the worked reference for everything above: about
fifteen lines, no framework import, and it runs under both samplers. Read it before writing
your own rule. Address it as `create_verifier("rmc")`, and pair it with `DraftTree.chain(L)` —
`max_children = 1` means `check_topology` refuses anything wider at construction time.

Its shape, in the coordinates of the section above: project the single child to `s_hat`;
accept with probability `1 ^ phi(s_hat - delta) / phi(s_hat)`; on rejection reflect about the
crossing point of the two densities, `s = delta - s_hat`, carrying the child's `z_perp`
through untouched; reconstruct. Per-step acceptance is `2 * Phi_bar(delta / 2)`, eq. (16) —
which is the overlap `1 - TV(P, Q)`, so it is also the ceiling for *any* single-proposal
coupling. Report `proposals_examined = 1`.

Two numerical points in that body worth stealing:

- **Never form `phi` itself.** The ratio simplifies to `exp(delta * (s_hat - delta/2))` — the
  quadratics cancel, the normalising constant cancels, and you avoid differencing two large
  nearly equal squared norms. This is why `specdiff.ops` ships `Phi` and `Phi_bar` but no
  Gaussian density: neither of the paper's rules needs one.
- **Compare in log space with `math.log1p(-u)`, not `math.log(u)`.** `ops.uniform` returns
  `[0, 1)`, so `u` can be exactly `0` and `math.log` raises `ValueError` — about once in
  `2^53` nodes on NumPy — but `torch.rand` is float32, so on the torch backend it is `2^-24`,
  about `6e-8`, which a long run will reach. `1 - u` is
  uniform too, and `log1p` is defined on precisely the range `uniform()` guarantees. No cap on
  the ratio is then needed: the left side is `<= 0`, so a positive log-ratio accepts
  unconditionally, which *is* the `1 ^ ·`. (In PyTorch `torch.log(0)` returns `-inf` rather
  than raising, so a rule ported from a tensor implementation can hide this bug.)

**Algorithm 2 — greedy rejection sampling, any `K`** — is `create_verifier("d-grs")`, and is
the rule a branching tree exists for. It sweeps the children *in drafting order* — this is the
sequence coupling, not the list coupling. It maintains the level `lambda_k` and the residual
mass `G_{k+1}`, accepting child `k` with probability
`1 ^ (rho(s_k) - lambda_{k-1})_+ / G_k` where `rho(s) = phi(s - delta) / phi(s)`. On rejection
the level rises to `lambda_k = lambda_{k-1} + G_k`, inducing the super-level set
`H_k = {s : rho(s) >= lambda_k}` and leaving `G_{k+1} = Q(H_k) - lambda_k P(H_k)`. After a full
sweep of rejections it samples the normalised residual (eq. 13). It reports
`proposals_examined = k` on acceptance and `K + 1` on the residual branch.

The masses are half-space masses in the projected coordinate —
`Q(H_k) = Phi_bar(tau_k - delta/2)` and `P(H_k) = Phi_bar(tau_k + delta/2)` with
`tau_k = ln(lambda_k) / delta` (Appendix B.2) — so no numerical integration is needed
*anywhere*, including in the residual: eq. (13)'s positive part is a half-line, so its CDF is
a difference of `Phi_bar`s and inverting it is a bisection.

Three things in that body that generalise to any sequence coupling:

- **Project every child before the sweep starts.** The residual branch needs `Z_perp` of the
  *first* child (line 18), not the last one examined, so the projections cannot be consumed
  and discarded as you go.
- **Accept on `u < beta`, not `u <= beta`.** `ops.uniform` returns `[0, 1)`, so `u` can be
  exactly `0`, and `<=` would accept even at `beta = 0` — which is the state *every child below
  the current level* is in, not a rare one. On the torch backend `torch.rand` is float32, so
  that misfires with probability `2^-24` (~`6e-8`) per such child; at a million of them it is a
  6% chance of one silently wrong acceptance.
- **Clamp `G_k` at zero.** It is a difference of two survival functions and can go a few ulps
  negative once the remaining mass is near zero; a negative mass turns the next `beta` into
  garbage rather than into a rejection.

Acceptance is `1 - G_{K+1}` (Theorem 2, eqs. 14–15) and rises with `K` — which is the whole
argument for a tree. At `K = 1` the recursion gives `lambda_1 = 1` and `G_2 = 2 Phi(delta/2) - 1`,
so it collapses to eq. (16): the same acceptance as RMC. The two rules differ at `K = 1` only
in what they return on rejection — RMC reflects, D-GRS draws from the residual.

`specdiff.ops` provides `standard_normal_cdf` and `standard_normal_sf` so neither needs SciPy,
and `Rank1Frame.tau` computes `tau_k` with the degeneracy guard already applied.

### Three traps

The first two are what makes Algorithm 2 harder than Algorithm 1 rather than merely longer:
at `K = 1` there is no order to get wrong and only one residual to carry, so both come for
free. The third bites either rule.

1. **Order.** The children arrive in sampling order and a sequence coupling must keep it.
   Sorting them or examining them by likelihood ratio breaks exactness.
2. **Which orthogonal residual you carry through.** Algorithm 2 returns `Z_perp,k` on
   acceptance of child `k` but `Z_perp,1` on the residual branch (lines 9 and 18). Using the
   wrong one produces samples that pass a casual eyeball check and fail `check_exactness`.
3. **Degeneracy.** Check `frame.degenerate` before computing any `tau`. See
   [above](#degeneracy).

`tests/test_rmc.py` and `tests/test_dgrs.py` are the shape your own rule's tests should take:
an exactness sweep over `delta` (including `0`) and `K`, the analytic acceptance probability
checked against the measured one, the structural claim of the rejection branch, and the
topology guard. Note that exactness alone is a weak test — `ResampleVerifier` passes every KS
check and accepts nothing — so the acceptance-probability assertion is what actually pins a
rule to the algorithm it claims to implement.

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
