# Plugging in your model

What you have to supply to sample from your own diffusion model, and what the library
supplies for you.

Two objects are mandatory — a `TargetTransition` and a `NoiseSchedule`. Everything else has a
usable default.

## The contract: eq. (24)

```
P_n(. | y) = N(m^p_n(y), sigma_n^2 I)     proposal
Q_n(. | y) = N(m^q_n(y), sigma_n^2 I)     target
```

Both kernels must be isotropic Gaussians that **share the variance schedule** and differ only
in their means. This is an assumption of the template, not an implementation detail — it is
what makes the rank-1 reduction legal, and a rule receiving a `VerifyRequest` is entitled to
rely on it. If your sampler does not fit this shape, it does not fit this library.

Anything that can produce those means plugs in: a real denoiser, a distilled draft network, an
analytic score, or the paper's delayed reverse drift.

## The target

Override `means`; **call the instance**, not `means` — `__call__` keeps the NFE accounting the
whole cost metric is defined on.

```python
from specdiff import TargetTransition

class MyDenoiser(TargetTransition):
    def __init__(self, net, times, gamma):
        super().__init__()                    # sets up the counters
        self.net, self.times, self.gamma = net, times, gamma

    def means(self, states, steps):
        """(rows, *state_shape) states at `rows` step indices -> means, same shape."""
        t = self.times[list(steps)]           # steps is a tuple of ints, one per row
        drift = self.net(states, t)
        return states + self.gamma * drift
```

`means` is called **once per round** with the round's entire verification batch — every
internal node of the draft tree, and under the batched sampler every live trajectory's
internal nodes too. It must be a single batched evaluation of the network. Batching it is the
whole point; a loop over rows throws the speedup away.

`steps` is a tuple of plain Python ints, and rows may sit at **different** steps. If your
network takes a scalar timestep, group by step:

```python
    def means(self, states, steps):
        out = np.empty_like(states)
        for step in sorted(set(steps)):
            idx = [i for i, s in enumerate(steps) if s == step]
            out[idx] = self._one_step(states[idx], step)
        return out
```

That is what `examples/gaussian_mixture.py` does. It costs one network call per *distinct*
step rather than one per row, and within a round the distinct steps number at most `L`.

`__call__` checks that you returned one row per input step, so a broadcasting slip fails
immediately rather than corrupting a trajectory.

## The schedule

```python
from specdiff import NoiseSchedule, ConstantSchedule, TabulatedSchedule

schedule = TabulatedSchedule(sigmas)          # pre-computed, e.g. eq. (37) for SD3
schedule = ConstantSchedule(0.2)              # toy models and tests

class MySchedule(NoiseSchedule):              # or compute it
    def sigma(self, step: int) -> float:
        return ...
```

**Indexing.** A round starting at step `n` uses scales `sigma_n .. sigma_{n + L_n - 1}`, and
the largest index that ever reaches the schedule is `N - 1`. A `TabulatedSchedule` therefore
needs exactly `N` entries, not `N + 1`.

**Zero is refused.** `NoiseSchedule.__call__` raises on a non-positive scale: at zero churn
both kernels are point masses, their TV distance is 1, and speculation is vacuous (Remark 3).
If your schedule ends at `sigma = 0`, stop the speculative sampler one step short and take the
final step with your own deterministic update.

### Converting an existing sampler

If you already have an Euler–Maruyama loop of the form

```
y_{n+1} = y_n + gamma * drift(y_n, t_n) + sigma_n * xi
```

then `m^q_n(y) = y + gamma * drift(y, t_n)` is your target mean and `sigma_n` your schedule —
that is the entire port. `examples/gaussian_mixture.py` does exactly this for a Gaussian
mixture, where the reverse drift is available in closed form (eqs. 32 and 35, discretised per
eqs. 4–5) and no network is involved, which makes it a good place to check your wiring before
pointing the sampler at a real model.

## The proposal

The cheap mean map `m^p`. Called once per tree level while drafting.

| class | `m^p(y)` | use |
| --- | --- | --- |
| `DelayedDriftProposal` | `y + (m^q(Y~) - Y~)` | **the paper's, eq. (7)** — self-speculative, no draft network needed |
| `IdentityProposal` | `y` | the worst useful proposal, and free — a floor for measuring proposal quality |
| `MirrorProposal` | `m^q(y)` | perfect (`delta = 0`, everything accepts). Useless in production, invaluable in tests |
| your own | anything | a distilled draft network is the obvious one |

`DelayedDriftProposal` is the default choice and needs no second model. The target mean is
`m^q_n(y) = y + gamma b^q(y)`, so the increment can be read off a mean the sampler already
computed:

```
gamma b^q(Y~) = m^q(Y~) - Y~
```

The proposal freezes that increment and reuses it at every depth of the tree. Because it is
read off a target mean that **verification already paid for** in an earlier round, no extra
target call is needed per round — this is Appendix C's root-drift prefetching. Exactly one
warm-up call is unavoidable at `n = 0`, and it is counted in `target_calls`.

```python
DelayedDriftProposal(target, prefetch=True)    # the paper's default
DelayedDriftProposal(target, prefetch=False)   # re-evaluate at each round's root:
                                               # +1 NFE per round, strictly better proposal
```

`prefetch=False` is a diagnostic, not a production setting: it isolates how much of your
acceptance rate is lost to drift staleness rather than to the proposal being weak.

### Writing your own

```python
from specdiff import ProposalTransition

class DraftNetProposal(ProposalTransition):
    def __init__(self, small_net):
        self.net = small_net

    def means(self, states, steps):
        return states + self.net(states, steps)
```

Three optional hooks exist because the interesting proposals are stateful:
`on_round_start(step, root_state)`, `on_verified(step, state, target_mean)`, and `reset()`.
A distilled draft network needs none of them and inherits the no-ops.

**If your proposal caches anything across calls, set `stateful = True`.** That is what stops a
single-trajectory proposal from being silently shared across a batch, where one cached drift
would be reused by trajectories sitting at different steps.

## Batching over images

The batched sampler needs a `BatchedProposal`, because a proposal now has to know which
trajectory each row belongs to:

| your proposal | wrap it in |
| --- | --- |
| stateless (draft net, `Identity`, `Mirror`) | `StatelessBatchedProposal(inner)` |
| the delayed drift | `BatchedDelayedDriftProposal(target)` — native, one `(batch, *shape)` buffer |
| any other stateful one | `PerSlotProposal(factory)` — one instance per trajectory |

```python
from specdiff import BatchedSpeculativeSampler, BatchedDelayedDriftProposal

sampler = BatchedSpeculativeSampler(
    target=my_target,
    proposal=BatchedDelayedDriftProposal(my_target),
    schedule=my_schedule,
    tree=DraftTree.uniform(branching=4, lookahead=3),
    verifier=my_rule,                 # unchanged — rules need no batching work
    num_steps=100,
    keep_trajectories=False,          # True keeps (batch, N+1, *shape); costs memory
)
result = sampler.sample(y0_batch)     # (batch, *state_shape)
```

`BatchedDelayedDriftProposal` collects the warm-up evaluations that trajectories need before
they hold any drift into **one** batched target call, not one per trajectory. `PerSlotProposal`
cannot — its instances are independent, so it costs one warm-up call per slot. Prefer a native
batched implementation when the proposal is hot.

Passing a `stateful` proposal to `StatelessBatchedProposal` raises rather than corrupting it.

Two further requirements: the tree must be level-uniform, and — see below — states must be
floating point.

## Choosing a tree

```python
DraftTree.uniform(branching=K, lookahead=L)   # the paper's (K, L) family
DraftTree.chain(L)                            # RMC's topology, K = 1
DraftTree.from_widths([3, 2, 2])              # depth-dependent widths
DraftTree.largest_uniform(budget=100, branching=4)   # deepest tree fitting a budget
DraftTree([-1, 0, 0, 1, 1, 2])                # arbitrary pruned tree, parents[u] < u
```

The cost knobs, for a uniform tree:

- `tree.budget` = `B` = `K + ... + K^L`, the proposal evaluations per round (eq. 12)
- `len(tree.internal_nodes)` = `|I|` = `B / K`, the **rows in the single target call** (eq. 26)
- `tree.depth` = `L`, the most steps one round can commit

Deeper trees raise the ceiling; wider ones raise the per-level acceptance probability. Both
cost target *batch volume* rather than target *calls*, so the trade is against memory and
per-call latency rather than against NFEs. `largest_uniform` picks the deepest tree that fits
a proposal budget.

The batched sampler additionally requires **level-uniform** trees — every node at a given depth
has the same width, so a level's candidates form a rectangular `(batch, K, *shape)` array.
`uniform` and `from_widths` qualify; arbitrary pruned trees do not, and are rejected at
construction with a message that says so.

## dtype and devices

**States must be floating point.** `float32` and `float64` both work; an integer array is
rejected outright:

```
TypeError: init has non-floating dtype dtype('int64'). Diffusion states must be
floating point: an integer array truncates every Gaussian draw to zero and the
sampler would return a silently wrong trajectory. Cast with e.g. `init.astype(float)`.
```

The guard exists because the failure it prevents is silent — an integer state truncates every
noise draw towards zero, and the run completes and reports a speedup while returning zeros.

The backend is resolved from the array you pass to `sample()`, and every intermediate inherits
its dtype and device. Under PyTorch that means a CUDA `init` keeps the whole run on device;
pass a `torch.Generator` on the same device as `rng`.

## Adding a backend

`ops.py` is the only module that touches NumPy or PyTorch. Porting to JAX, MLX or anything else
is one `Backend` subclass and no other edits:

```python
from specdiff.ops import Backend

class JaxBackend(Backend):
    name = "jax"
    # 17 methods: zeros_stack, randn_stack, uniform, make_rng, take, put,
    # repeat_rows, stack_rows, scale_rows, group_rows, norm, dot, copy,
    # is_finite, is_floating, finfo_eps, allclose

sampler = SpeculativeSampler(..., backend=JaxBackend())
```

State arrays are always *stacks* of shape `(num_nodes, *state_shape)`. Node and step indices
are plain Python ints; nothing framework-specific crosses the public API except the state
arrays themselves.

`resolve_backend` picks NumPy or PyTorch automatically from the type of the array you pass in,
so an explicit `backend=` is only needed for a framework it does not know.

## Sanity checks before you trust a run

```python
from specdiff import standard_sampler

# 1. Does the reference loop reproduce your existing sampler's output distribution?
ref = standard_sampler(my_target, my_schedule, num_steps=N)
print(ref.sample(y0, rng=rng).summary())     # must report exactly 1.000x

# 2. Does a perfect proposal accept everything?
#    MirrorProposal gives delta = 0, so any exact rule accepts at every level.

# 3. What is the mismatch on your model, before you write a coupling?
#    See docs/writing-a-verifier.md -> "Measuring your headroom first".
```

`standard_sampler` is `K = L = 1` with a rule that always resamples: one committed step per
target call, i.e. `N` NFEs, which is a plain Euler–Maruyama loop. If Algorithm 3 with that rule
does not match your existing sampler in distribution, the bug is in the wiring, not in the
coupling.
