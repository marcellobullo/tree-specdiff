# PAWS List Coupling

PAWS is a verifier, not a separate sampler. Use the existing draft tree, proposal,
schedule, batching, prefetch, and Picard refinement machinery:
```python
from specdiff import DraftTree, RankSelectionCoupling, create_verifier

tree = DraftTree.uniform(branching=2, lookahead=3)
verifier = create_verifier("paws")
# Equivalent: RankSelectionCoupling(rank_policy="optimized", residual_complement="first")
# Pass tree and verifier to SpeculativeSampler or BatchedSpeculativeSampler.
```

The registration name refers to the rank-selection variant in the sibling
`accelerating-diffusion-sampling` repository, not to Algorithms 1 or 2 of the
specdiff paper. SciPy is a required dependency. There is no temperature parameter.

## Why the coupling works

Condition on the current state and any information used to construct the proposal
mean. The outgoing children must be conditionally iid
$X_i\sim p=\mathcal N(m_p,\sigma^2I)$, and the target transition must be
$q=\mathcal N(m_q,\sigma^2I)$, with shared $\sigma>0$.

Let $e=(m_q-m_p)/\|m_q-m_p\|$ and
$\delta=\|m_q-m_p\|/\sigma$. Write
$X_i=m_p+\sigma(S_ie+W_i)$. The $S_i$ are iid standard normals;
the Gaussian orthogonal components $W_i$ are independent of every projection.
Under the target, only the scalar law changes, to $S\sim\mathcal N(\delta,1)$.

Sort $U_i=\Phi(S_i)$. Choose a rank independently of the realized list, using
probabilities $\lambda_1,\ldots,\lambda_K$. Its uniform-coordinate density is

$$
\beta_\lambda(u)=\sum_{r=1}^K\lambda_r
  \frac{K!}{(r-1)!(K-r)!}u^{r-1}(1-u)^{K-r}.
$$

The selected projection has density
$g(s)=\phi(s)\beta_\lambda(\Phi(s))$, while the target has
$f(s)=\phi(s-\delta)$. Accept the selected child with probability
$\min(1,f(s)/g(s))$. The accepted output has unnormalized density
$\min(f,g)$, and total acceptance probability
$A=\int\min(f,g)$.

On rejection, sample a projection from
$(f-g)_+/(1-A)$. Then the total scalar output density is exactly

$$
\min(f,g)+(f-g)_+=f.
$$

The decision uses only scalar projections and independent randomness, leaving
the selected child's orthogonal Gaussian component unchanged in law. On
rejection, each complement policy below has that same independence property.
Thus the full state has distribution $q$.

An accepted output is the actual child at its original, unsorted index, so the
sampler can descend into its subtree. A rejected output is corrected and ends
the round. Repeating this conditional argument gives the target Markov
trajectory, not merely the right terminal marginal. The existing causal Picard
update preserves the required conditional Gaussian child law; proposal
construction must never inspect its own outgoing innovations.

This is maximal coupling of the *rank-selected density* with the target.
Optimizing over rank weights is not the same as optimizing over every possible
list-selection rule, and does not establish dominance over D-GRS.

## Independent variant choices

| Option | Choices | Meaning |
| --- | --- | --- |
| `rank_policy` | `optimized` (default) | Cached linear-program approximation to maximum overlap |
| | `uniform` | Equal rank probabilities; selected density is the original proposal |
| | `max` | Largest projection toward the target mean; not necessarily best at small gaps |
| | callable `(delta, K) -> weights` | Finite, nonnegative weights with positive sum, normalized internally |
| `residual_complement` | `first` (default) | Reuse the first child's orthogonal component, as D-GRS does |
| | `fresh` | Draw independent Gaussian noise and remove its projection |
| | `nearest_projection` | Reuse the component of the child closest to the corrected scalar |

A callable rank policy must not depend on the realized children, orthogonal
components, or future subtree information. Nearest *full-vector* selection is
not an interchangeable complement policy: it generally biases the orthogonal
component. `first` means first in drafting order, not first after sorting.

For `K=1` (or uniform ranks for any `K`), acceptance is
$2\Phi(-\delta/2)$. This agrees with RMC's acceptance probability, but the
rejection coupling and random-number consumption differ.

## Numerical implementation

Acceptance is evaluated in log space with the **actual** normalized mean gap,
using stable log-normal-CDF and log-survival values. The optimizer alone rounds
the gap to three significant digits for its bounded cache. Its finite-grid LP
is approximate; uniform and maximum-rank policies are retained as candidates.
Extremely large gaps use maximum rank directly. These choices affect efficiency,
not the correction formula.

The residual uses **inverse-CDF sampling with a closed-form CDF**. There is no
numerical integration. Both laws have exact CDFs, so the residual mass on any
interval where $f>g$ is arithmetic:

$$
\text{mass}(a,b)=\bigl(T(a)-T(b)\bigr)-\bigl(D(b)-D(a)\bigr),
\qquad T(x)=\Phi(x)-\Phi(x-\delta).
$$

$\beta_\lambda$ is a degree $K-1$ polynomial in $u=\Phi(s)$, being a
mixture of Beta order-statistic densities with integer parameters. Its integral
is therefore the degree $K$ Bernstein polynomial whose coefficients are the
**cumulative** rank weights $\Lambda_j=\lambda_1+\cdots+\lambda_j$:

$$
P(S_{\text{sel}}\le s)=\sum_{j=1}^{K}\Lambda_j\binom{K}{j}u^j(1-u)^{K-j}.
$$

Two deviation forms keep the arithmetic *relative* rather than absolute, which
is what makes small gaps work. $\sum_j (j/K)\binom{K}{j}u^j(1-u)^{K-j}=u$
exactly, so $D$, the selected CDF's deviation from $\Phi$, uses
coefficients $\Lambda_j-j/K$; these vanish identically for uniform weights.
And $T$ is computed from the midpoint series
$\delta\,\phi(m)\bigl(1+h^2He_2(m)/6+\cdots\bigr)$, $m=x-\delta/2$,
$h=\delta/2$, rather than by differencing two normal CDFs, which loses every
digit as $\delta\to0$. Every term is then the size of the residual, not the
size of a CDF. Both $T$ and $D$ vanish at $\pm\infty$, so the two
unbounded spans need no special case.

The only numerical step is locating the crossings of $f-g$, by a vectorised
sign sweep refined with Brent. That step is well conditioned in exactly the way
quadrature was not: $f-g$ vanishes **linearly** at a crossing, so an error
$\varepsilon$ in a crossing location costs only $O(\varepsilon^2)$ of
mass. Support may be disconnected -- two spans is the normal case, and
maximum-rank selection puts one in each tail -- so spans are found, not assumed.
Residual caches use the exact gap and weights, not the optimizer's rounded key.

This is still numerical, not symbolic: floating-point arithmetic, the accuracy
of `ndtr`/`log_ndtr` in the far tails, crossing location, Brent inversion
(`xtol=2e-13`), and the shared `Rank1Frame` degeneracy tolerance
(`delta <= 1e-10` by default) qualify claims of exactness. For the small-gap
shortcut alone, the total-variation discrepancy is approximately
$\delta/\sqrt{2\pi}$.

The sibling PAWS code instead uses direct residual rejection from the target.
That takes on average $1/(1-A)$ trials **conditional on needing a correction**,
which can be large when acceptance is high. But correction itself occurs with
probability $1-A$, so a large conditional retry count does not imply huge
amortized cost in every regime. Inverse-CDF sampling removes this random retry
loop and, with a closed-form CDF, adds only root-finding work: 0.34 ms per
correction against 4.75 ms for the adaptive-quadrature version it replaced.

Scalar numerical work runs on CPU; states remain on their original backend.
Batched verification currently uses the generic row-wise verifier interface.
GPU scalar synchronization and numerical inversion can matter for throughput.
Benchmark wall time and target rows as well as sequential target calls before
claiming acceleration. Tree memory still grows as $K+\cdots+K^L$.
`proposals_examined` counts all `K` projections used by list selection
(except the equal-kernel shortcut, which uses one child).

## Experiments

GM and Picard sweeps include `paws` in their default rule lists. Image drivers
accept `--rule paws`; their sweep scripts also include PAWS. All use verifier
capabilities (`requires_chain`, `matched_tree`) to allocate trees or matched
chains. Image adapters handle deterministic endpoints outside speculation;
passing zero noise directly to PAWS is an error.

Variants use a per-rule JSON mapping, so PAWS options are not passed to RMC or
D-GRS:
```bash
python experiments/gm/gm_sweep.py --out results/paws-smoke \
  --rules paws --dimension 8 --num-steps 8 --K-values 2 --L-values 2 \
  --replicates 4 --n-workers 1 \
  --verifier-options '{"paws":{"rank_policy":"max","residual_complement":"first"}}'

python experiments/images/run_edm.py --toy --no-accelerate --device cpu \
  --rule paws --branching 2 --lookahead 2 --num-steps 8 --num-samples 4 \
  --out results/paws-edm-smoke
```

The same `--verifier-options` flag is supported by `picard_sweep.py` and
`run_sd3.py`. In image run-config files, `method.verifier_options` is a JSON
**string** containing that mapping. Image shell sweeps accept the
`VERIFIER_OPTIONS` environment variable. Run signatures/configs record these
options, and resume checks reject incompatible settings. Use separate output
directories for separate variants; plotting labels the rule as PAWS, not as a
distinct series for every option combination.

Tests cover scalar target laws, joint orthogonal laws, disconnected residual
quantiles, single-child acceptance, optimizer efficiency, original child
indexing, zero uniforms, tiny absolute gaps, CPU Torch dtypes, batching,
truncated/nonuniform trees, Picard trajectories, and experiment wiring.
`tests/test_paws_residual.py` covers the closed form specifically: the gaps from
the reported eps=0.1 sweep cells, quantile monotonicity, crossing-scan
completeness, and both deviation forms. It checks against two oracles, because
neither alone suffices -- quadrature split at the crossings validates the
formula where the density is well scaled, and arbitrary precision (`mpmath`,
skipped if absent) validates the arithmetic everywhere, including a span near
$s=-29$ whose mass every float64 intermediate underflows.
