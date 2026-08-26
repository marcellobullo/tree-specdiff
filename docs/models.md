# Integrating a model

This guide describes the components required to sample from a diffusion model.

A `TargetTransition` and a `NoiseSchedule` are required. The remaining components provide
default implementations.

## Transition contract: Equation (24)

```
P_n(. | y) = N(m^p_n(y), sigma_n^2 I)     proposal
Q_n(. | y) = N(m^q_n(y), sigma_n^2 I)     target
```

Both kernels must be isotropic Gaussians that **share the variance schedule** and differ only
in their means. This structural requirement enables the rank-1 reduction, and verifiers may
rely on it. Samplers with different covariance structures are not supported.

Any implementation that produces these means can be used, including a denoiser, distilled
draft network, analytic score, or delayed reverse drift.

## The target

Override `means`, but invoke the target instance through `__call__`, which maintains NFE
accounting.

```python
from specdiff import TargetTransition

class MyDenoiser(TargetTransition):
    def __init__(self, net, times, gamma):
        super().__init__()                    # sets up the counters
        self.net, self.times, self.gamma = net, times, gamma

    def means(self, indices_in_batch, states, steps):
        """(rows, *state_shape) states at `rows` step indices -> means, same shape."""
        t = self.times[list(steps)]           # steps is a tuple of ints, one per row
        drift = self.net(states, t)
        return states + self.gamma * drift
```

`means` is called **once per round** with the round's entire verification batch — every
internal node of the draft tree, and under the batched sampler every live trajectory's
internal nodes too. Implement it as one batched network evaluation; evaluating rows
individually removes the intended parallelism.

`steps` is a tuple of plain Python ints, and rows may sit at **different** steps. If your
network takes a scalar timestep, group by step:

```python
    def means(self, indices_in_batch, states, steps):
        out = np.empty_like(states)
        for step in sorted(set(steps)):
            idx = [i for i, s in enumerate(steps) if s == step]
            out[idx] = self._one_step(states[idx], step)
        return out
```

That is what `examples/gaussian_mixture.py` does. It costs one network call per *distinct*
step rather than one per row, and within a round the distinct steps number at most `L`.

`__call__` validates that the result contains one row per input step, so shape errors fail
before sampling continues.

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

**Scales must be positive.** `NoiseSchedule.__call__` raises on a non-positive scale. At zero
churn, both kernels are point masses, their total-variation distance is 1, and speculation
provides no benefit (Remark 3).
If your schedule ends at `sigma = 0`, stop the speculative sampler one step short and take the
final step with your own deterministic update.

### Converting an existing sampler

If you already have an Euler–Maruyama loop of the form

```
y_{n+1} = y_n + gamma * drift(y_n, t_n) + sigma_n * xi
```

then `m^q_n(y) = y + gamma * drift(y, t_n)` is the target mean and `sigma_n` is the schedule.
`examples/gaussian_mixture.py` applies this conversion to a Gaussian
mixture, where the reverse drift is available in closed form (eqs. 32 and 35, discretised per
Equations 4–5). Because the example requires no network, it is a useful integration check
before using a learned model.

## The proposal

The cheap mean map `m^p`. Called once per tree level while drafting.

| class | `m^p(y)` | use |
| --- | --- | --- |
| `DelayedDriftProposal` | `y + (m^q(Y~) - Y~)` | **the paper's, eq. (7)** — self-speculative, no draft network needed |
| `IdentityProposal` | `y` | zero-cost baseline for measuring proposal quality |
| `MirrorProposal` | `m^q(y)` | ideal proposal (`delta = 0`) for tests and diagnostics |
| custom | application-defined | for example, a distilled draft network |

`DelayedDriftProposal` is the default choice and needs no second model. The target mean is
`m^q_n(y) = y + gamma b^q(y)`, so the increment can be read off a mean the sampler already
computed:

```
gamma b^q(Y~) = m^q(Y~) - Y~
```

The proposal freezes that increment and reuses it at every depth of the tree. Because it is
read from a target mean already evaluated during verification, no additional target call is
needed per round. This is Appendix C's root-drift prefetching. One warm-up call is required at
`n = 0` and included in `target_calls`.

```python
DelayedDriftProposal(target, prefetch=True)    # the paper's default
DelayedDriftProposal(target, prefetch=False)   # re-evaluate at each round's root:
                                               # +1 NFE per round, strictly better proposal
```

Use `prefetch=False` for diagnostics. It measures how much acceptance loss is attributable to
drift staleness instead of proposal error.

### Writing your own

```python
from specdiff import ProposalTransition

class DraftNetProposal(ProposalTransition):
    def __init__(self, small_net):
        self.net = small_net

    def means(self, indices_in_batch, states, steps):
        return states + self.net(states, steps)
```

`indices_in_batch[i]` says which of the `batch_size` images entry `i` belongs to. A proposal
that keeps no per-image memory — a draft network, `Identity`, `Mirror` — ignores it, as above.

Three optional hooks support proposals with per-image state:
`on_round_start(indices_in_batch, steps, roots)`,
`on_verified(indices_in_batch, steps, states, target_means)`, and `reset(batch_size)`.
A distilled draft network needs none of them and inherits the no-ops.

**Key cached proposal state by `indices_in_batch`.** Store it in a
`(batch_size, *state_shape)` buffer allocated by `reset`, as `DelayedDriftProposal` does. This
prevents state from being shared between trajectories at different steps.

## Batching over images

Proposals already receive `indices_in_batch`, so the same object works with scalar and batched
samplers. A scalar sample is represented by `batch_size = 1`.

```python
from specdiff import BatchedSpeculativeSampler, DelayedDriftProposal

sampler = BatchedSpeculativeSampler(
    target=my_target,
    proposal=DelayedDriftProposal(my_target),
    schedule=my_schedule,
    tree=DraftTree.uniform(branching=4, lookahead=3),
    verifier=my_rule,                 # unchanged — rules need no batching work
    num_steps=100,
    keep_trajectories=False,          # True keeps (batch, N+1, *shape); costs memory
)
result = sampler.sample(y0_batch)     # (batch, *state_shape)
```

`DelayedDriftProposal` collects the warm-up evaluations that images need before they hold any
drift into one batched target call and indexes its drift buffer by `indices_in_batch` to keep
trajectory state isolated.

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

**States must use a floating-point dtype.** Both `float32` and `float64` are supported; integer
arrays are rejected:

```
TypeError: init has non-floating dtype dtype('int64'). Diffusion states must be
floating point: an integer array truncates every Gaussian draw to zero and the
sampler would return a silently wrong trajectory. Cast with e.g. `init.astype(float)`.
```

Integer states truncate noise draws toward zero and invalidate the result. The dtype check
prevents sampling from continuing in that state.

The backend is resolved from the array you pass to `sample()`, and every intermediate inherits
its dtype and device. Under PyTorch that means a CUDA `init` keeps the whole run on device;
pass a `torch.Generator` on the same device as `rng`.

## Adding a backend

`ops.py` contains the NumPy and PyTorch integration. Supporting another framework such as JAX
or MLX requires a new `Backend` subclass:

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

## Integration checks

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
does not match the existing sampler in distribution, inspect the model and schedule
integration before evaluating the coupling.
