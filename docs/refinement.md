# Proposal refinement

Proposal refinement applies synchronous Picard sweeps to an already drafted tree. It is
disabled by default and does not change the existing algorithm unless
proposal_refinement_iters is positive.

```python
from specdiff import SpeculativeSampler, picard_update_fn

sampler = SpeculativeSampler(
    target=target,
    proposal=proposal,
    schedule=schedule,
    tree=tree,
    verifier=verifier,
    num_steps=N,
    proposal_refinement_iters=2,
    refinement_update_fn=picard_update_fn,
)
```

The same arguments are accepted by BatchedSpeculativeSampler.

## Tree Picard recurrence

Consider a round beginning at committed state $Y_n$. Let $r$ be the tree root,
$\operatorname{pa}(v)$ the parent of $v$, and $|u|$ the depth of $u$. The transition
leaving $u$ has step

$$
s(u)=n+|u|.
$$

Sample one innovation for every non-root node and retain it throughout refinement:

$$
\xi_v\sim\mathcal N(0,I),\qquad
\eta_v=\sigma_{s(\operatorname{pa}(v))}\xi_v.
$$

The ordinary proposal draft is iteration zero:

$$
X_r^{(0)}=Y_n,\qquad
X_v^{(0)}=m^p_{s(u)}(X_u^{(0)})+\eta_v,
\quad u=\operatorname{pa}(v).
$$

For sweep $j$, the built-in target-backed update evaluates all internal nodes in one
batch:

$$
G_u^{(j)}
=m^q_{s(u)}(X_u^{(j)})-X_u^{(j)}.
$$

It then reconstructs the tree breadth-first:

$$
X_r^{(j+1)}=Y_n,
$$

$$
\widetilde\mu_u^{(j+1)}
=X_u^{(j+1)}+G_u^{(j)},
$$

$$
X_v^{(j+1)}
=\widetilde\mu_u^{(j+1)}+\eta_v.
$$

The mixed iteration indices are essential: the new parent anchors the path sum, while the
increment was evaluated at the frozen old parent. Along the unique path
$r=u_0,\ldots,u_\ell=v$,

$$
X_v^{(j+1)}
=Y_n+\sum_{i=0}^{\ell-1}
\left[
m^q_{n+i}(X_{u_i}^{(j)})-X_{u_i}^{(j)}
+\eta_{u_{i+1}}
\right].
$$

This is the tree analogue of ParaDiGMS's prefix sum: shared path prefixes are computed once,
and all expensive target rows in one sweep are evaluated in parallel.

## Conditional proposal law

After $J\ge1$ sweeps, the actual proposal mean stored for parent $u$ is

$$
\widetilde\mu_u^{(J)}
=
X_u^{(J)}
+m^q_{s(u)}(X_u^{(J-1)})
-X_u^{(J-1)}.
$$

Let $\mathcal A_u$ contain the committed root, fixed model conditioning, and innovations
on the path from the root to $u$, but not the outgoing innovations of $u$. Then
$\widetilde\mu_u^{(J)}$ is $\mathcal A_u$-measurable and

$$
X_v^{(J)}\mid\mathcal A_u
\overset{\mathrm{i.i.d.}}{\sim}
\mathcal N\left(\widetilde\mu_u^{(J)},\sigma_{s(u)}^2I\right),
\qquad v\in C(u).
$$

The marginal node law is generally a Gaussian mixture when the drift is nonlinear. The
local conditional Gaussian law is what RMC and D-GRS require.

Final verification compares

$$
P_u^{(J)}
=\mathcal N(\widetilde\mu_u^{(J)},\sigma_s^2I)
$$

with

$$
Q_u^{(J)}
=\mathcal N(m_s^q(X_u^{(J)}),\sigma_s^2I).
$$

For RMC, conditionally on $\mathcal A_u$, the normalized displacement and acceptance are

$$
\Delta_u^{(J)}
=
\frac{
[m_s^q(X_u^{(J)})-X_u^{(J)}]
-
[m_s^q(X_u^{(J-1)})-X_u^{(J-1)}]
}{\sigma_s},
$$

$$
\Pr(\mathrm{accept}\mid\mathcal A_u)
=2\Phi\left(-\frac{\|\Delta_u^{(J)}\|}{2}\right).
$$

For $K>1$, D-GRS uses the same conditional proposal and target kernels but its
multi-proposal acceptance recursion.

## Finite-depth propagation

Let the exact target tree use the same retained innovations:

$$
X_r^\star=Y_n,\qquad
X_v^\star=m^q_{s(u)}(X_u^\star)+\eta_v.
$$

Nested induction over sweep and depth gives

$$
X_v^{(J)}=X_v^\star\qquad\text{for every }|v|\le J.
$$

No contraction assumption is needed for this finite causal statement. Each sweep propagates
exact target information at least one level deeper. Therefore, for a tree of depth $L$,

$$
J\ge L\Longrightarrow X^{(J)}=X^\star.
$$

The original proposal drift is a no-op refiner: $X^{(0)}$ already satisfies its recursive
proposal equation, so applying the same drift reproduces $X^{(0)}$.

## Callback contract

A callback receives one row for every internal node, potentially including ancestors and
descendants of one another. Output row \(i\) must be local:

$$
G_i^{(j)}
=f_j(b_i,u_i,s_i,X_i^{(j)},\widetilde\mu_i^{(j)},\sigma_i).
$$

It may use fixed parameters and conditioning for image $b_i$, but it must not inspect,
reduce over, or mix other rows. Cross-row dependence can make an ancestor's mean depend on
its outgoing noise and invalidate the conditional proposal law.

Callbacks always return:

```python
RefinementUpdate(
    increments=increments,
    exact_target_means=target_means_or_none,
)
```

`exact_target_means[i]`, when present, is a correctness-bearing assertion that it equals
the target mean at the request's exact image, step, and state. An incorrect assertion can
invalidate sampling. With check_contract=True, every reused assertion is checked in one
unaccounted target batch.

Target transitions and callbacks must be deterministic and row-independent for fixed image
conditioning, step, and state. Batch partitioning is an execution choice, not part of the
mathematical map.

## Exact final target-mean reuse

The final Picard sweep evaluates $m_s^q(X_u^{(J-1)})$, while verification needs
$m_s^q(X_u^{(J)})$. The cached value is reused only when all of the following match:

- target instance
- image index
- logical node and flat-buffer identity
- step
- represented state, using exact elementwise equality.

Thus

$$
X_u^{(J)}=X_u^{(J-1)}
\Longrightarrow
m_s^q(X_u^{(J)})=m_s^q(X_u^{(J-1)}).
$$

No tolerance is used. A custom backend that does not implement exact row comparison
conservatively re-evaluates every row. GPU nondeterminism can only reduce reuse; it cannot
cause approximate states to be treated as identical.

Without leaf evaluation, target-backed scalar refinement uses

$$
J+\mathbf 1\{J<L_n\}
$$

target calls per round, excluding proposal-owned target calls. For a batched iteration the
corresponding count is

$$
J+\mathbf 1\{\exists i:J<L_{n_i}\}.
$$

With evaluate_leaves=True, leaves are not refinement parents, so a final target call is
still needed whenever active leaves are requested.

Reuse between refinement sweeps is intentionally not implemented. It could reduce target
batch volume after shallow nodes stabilize, but normally not target-call depth while deeper
nodes are still changing.

## Accounting

Each round separates target work into:

- `proposal_target_calls` and `proposal_target_states_evaluated`;
- `refinement_target_calls` and `refinement_target_states_evaluated`;
- `verification_target_calls` and `verification_target_states_evaluated`;
- `verification_target_means_reused`.

The three call categories and state-count categories sum to the round totals. Global
`SamplingResult.target_calls` remains the speedup denominator.
