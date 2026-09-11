# Proposal refinement

Proposal refinement applies synchronous Picard sweeps to an already drafted tree, before
verification. It is disabled by default and does not change the algorithm unless
`proposal_refinement_iters` is positive.

```python
from specdiff import SpeculativeSampler

sampler = SpeculativeSampler(
    target=target,
    proposal=proposal,
    schedule=schedule,
    tree=tree,
    verifier=verifier,
    num_steps=N,
    proposal_refinement_iters=2,  # J sweeps per round
)
```

`BatchedSpeculativeSampler` accepts the same arguments. Positive iteration counts use
`picard_drift_update_fn`, which freezes only the target's drift ([§3](#3-the-two-built-in-updates)).
Pass `refinement_update_fn=picard_update_fn` to freeze the whole increment $m-x$ instead. That was
the default before, and it reproduces earlier runs. The GM Picard sweep exposes the same choice as
`--picard-update {drift,increment}`.

This page derives everything the implementation relies on:

1. [Setting and notation](#1-setting-and-notation)
2. [Picard sweeps as a split of the target mean](#2-picard-sweeps-as-a-split-of-the-target-mean)
3. [The two built-in updates](#3-the-two-built-in-updates)
4. [Path-sum form](#4-path-sum-form)
5. [Conditional proposal law and exactness](#5-conditional-proposal-law-and-exactness)
6. [Finite-depth propagation](#6-finite-depth-propagation)
7. [Mismatch at verification](#7-mismatch-at-verification)
8. [Size of the stale affine term](#8-size-of-the-stale-affine-term)
9. [Error propagation across sweeps](#9-error-propagation-across-sweeps)
10. [Floating point, target-mean reuse, and cost](#10-floating-point-target-mean-reuse-and-cost)
11. [Callback contract](#11-callback-contract)
12. [Accounting](#12-accounting)
13. [Measured effect on the Gaussian mixture](#13-measured-effect-on-the-gaussian-mixture)

## 1. Setting and notation

A round begins at the committed state $Y_n$. In the draft tree, $r$ is the root, $\operatorname{pa}(v)$
is the parent of $v$, $C(u)$ is the set of children of $u$, and $|u|$ is the depth of $u$. The
transition leaving $u$ has step

$$
s(u)=n+|u|.
$$

Proposal and target kernels share the variance schedule (Appendix C, eq. 24):

$$
P_s(\cdot\mid x)=\mathcal N\big(m^p_s(x),\,\sigma_s^2 I\big),
\qquad
Q_s(\cdot\mid x)=\mathcal N\big(m^q_s(x),\,\sigma_s^2 I\big).
$$

Phase 1 samples one innovation for every non-root node. Refinement keeps it fixed:

$$
\xi_v\sim\mathcal N(0,I)\ \text{i.i.d.},
\qquad
\eta_v=\sigma_{s(\operatorname{pa}(v))}\,\xi_v .
$$

The ordinary draft is iteration zero. The exact target tree uses the same innovations. With
$u=\operatorname{pa}(v)$:

$$
X_r^{(0)}=Y_n,\quad X_v^{(0)}=m^p_{s(u)}\big(X_u^{(0)}\big)+\eta_v;
\qquad
X_r^\star=Y_n,\quad X_v^\star=m^q_{s(u)}\big(X_u^\star\big)+\eta_v .
$$

$X_u^{(j)}$ is node $u$ after sweep $j$, and $\widetilde\mu_u^{(j)}$ is the proposal mean stored for
parent $u$ after sweep $j$, with $\widetilde\mu_u^{(0)}=m^p_{s(u)}(X_u^{(0)})$. Every node satisfies
$X_v^{(j)}=\widetilde\mu_{\operatorname{pa}(v)}^{(j)}+\eta_v$.

## 2. Picard sweeps as a split of the target mean

A sweep evaluates the target once, in one batch, at the frozen snapshot $X^{(j)}$ of all
internal nodes. It then rebuilds the tree breadth-first, so every parent is rebuilt before its
children:

$$
X_r^{(j+1)}=Y_n,
\qquad
\widetilde\mu_u^{(j+1)}=\mathcal M_{s(u)}\big(X_u^{(j)};\,X_u^{(j+1)}\big),
\qquad
X_v^{(j+1)}=\widetilde\mu_u^{(j+1)}+\eta_v .
$$

The update map $\mathcal M_s(\text{snapshot};\,\text{anchor})$ is where updates differ. The only
requirement is consistency on the diagonal:

$$
\mathcal M_s(x;\,x)=m^q_s(x)\quad\text{for every }x.
\qquad\text{(C)}
$$

Both built-in updates have the form

$$
\mathcal M_s(y;\,x)=\operatorname{apply}_s\big(D_s(y),\,x\big),
\qquad
D_s(y)=\operatorname{freeze}_s\big(y,\,m^q_s(y)\big),
$$

for a pair of maps with $\operatorname{apply}_s\big(\operatorname{freeze}_s(y,m),\,y\big)=m$, which gives
(C). The target is evaluated only at the snapshot $y$; the anchor $x$ enters through
$\operatorname{apply}$ alone, so the rebuild needs no further target calls.

When $\operatorname{apply}$ is additively separable, $\operatorname{apply}_s(D,x)=A_s(x)+B_s(D)$, the update
is a split of the target mean:

$$
m^q_s=A_s+N_s,
\qquad
N_s=B_s\circ D_s,
\qquad
\mathcal M_s(y;\,x)=A_s(x)+N_s(y).
$$

$A_s$ is evaluated at the new anchor and $N_s$ is frozen at the snapshot. The mixed iteration
indices are essential: the new parent anchors the step, while the frozen part comes from the old
parent. [§7](#7-mismatch-at-verification) and [§9](#9-error-propagation-across-sweeps) show that
only the frozen part produces mismatch and propagates error. The best split therefore puts
everything cheap and known in closed form into $A_s$, and freezes only the expensive network
output.

## 3. The two built-in updates

**Whole increment (`picard_update_fn`).** This uses the translation pair
$\operatorname{freeze}(y,m)=m-y$ and $\operatorname{apply}(D,x)=x+D$:

$$
A_s(x)=x,
\qquad
N_s(y)=G_s(y)=m^q_s(y)-y,
\qquad
\widetilde\mu_u^{(j+1)}=X_u^{(j+1)}+G_s\big(X_u^{(j)}\big).
$$

The callback returns $G_s(X_u^{(j)})$ as `increments`, and `refine_tree` adds them to the rebuilt
parents.

**Frozen drift (`picard_drift_update_fn`, the default).** This uses the target's own
`freeze_drift` and `apply_drift`, the hooks that `DelayedDriftProposal` already relies on (see
[models.md](models.md)). The callback returns $D_s(X_u^{(j)})$ as `drifts`, and `refine_tree` applies
them at the rebuilt parents.

The churn kernels of `experiments/` and `examples/gaussian_mixture.py` are affine in the network
velocity. Let $t=t_s$ be the interpolant level (the code's `sigma`, running from 1 to 0),
$h=h_s=t_s-t_{s+1}>0$ the step (the code's `dt` equals $-h$), $\varepsilon$ the churn, and
$g^2(t)=2t/(1-t)$. For the linear interpolant $x=(1-t)x_0+t\xi$ the score is

$$
\operatorname{score}_s(x)=-\frac{x+(1-t)\,v_s(x)}{t},
$$

and one churn step has mean

$$
m^q_s(x)=x-h\Big[v_s(x)-\tfrac12\varepsilon^2 g^2(t)\,\operatorname{score}_s(x)\Big].
$$

The churn correction splits into a linear part and a velocity part:

$$
-\tfrac12\varepsilon^2 g^2(t)\,\operatorname{score}_s(x)
=\frac{\varepsilon^2}{1-t}\,x+\varepsilon^2\,v_s(x).
$$

Collecting terms:

$$
m^q_s(x)=a_s\,x+b_s\,v_s(x),
\qquad
a_s=1-\frac{h\,\varepsilon^2}{1-t},
\qquad
b_s=-h\,(1+\varepsilon^2).
$$

This is `affine()` in `experiments/gm/models.py`. The deterministic Euler fallback has
$a_s=1$ and $b_s=-h$. The hooks freeze the velocity, $\operatorname{freeze}(y,m)=(m-a_s y)/b_s$ and
$\operatorname{apply}(D,x)=a_s x+b_s D$, so

$$
A_s(x)=a_s x,
\qquad
N_s(y)=b_s\,v_s(y),
\qquad
\widetilde\mu_u^{(j+1)}=a_s X_u^{(j+1)}+b_s\,v_s\big(X_u^{(j)}\big).
$$

For the same kernel, the whole-increment update freezes

$$
G_s(y)=(a_s-1)\,y+b_s\,v_s(y),
$$

which contains the linear piece $(a_s-1)y$. That piece costs nothing to evaluate at the new
anchor. For the churn kernels, $a_s\neq1$ exactly when $\varepsilon > 0$. For a target whose mean
is a translation of its input ($a_s=1$), the two updates coincide.

## 4. Path-sum form

Unrolling the rebuild along the path $r=u_0,\ldots,u_\ell=v$ writes every node through the
previous sweep only.

**Whole increment.** The rebuild gives
$X_{u_{i+1}}^{(j+1)}-X_{u_i}^{(j+1)}=G_{n+i}\big(X_{u_i}^{(j)}\big)+\eta_{u_{i+1}}$, which telescopes to

$$
X_v^{(j+1)}=Y_n+\sum_{i=0}^{\ell-1}\Big[G_{n+i}\big(X_{u_i}^{(j)}\big)+\eta_{u_{i+1}}\Big].
$$

This is the tree analogue of ParaDiGMS's prefix sum.

**Frozen drift, affine kernel.** The rebuild gives
$X_{u_{i+1}}^{(j+1)}=a_{n+i}X_{u_i}^{(j+1)}+b_{n+i}v_{n+i}\big(X_{u_i}^{(j)}\big)+\eta_{u_{i+1}}$. Induction on
$\ell$ gives the discounted sum

$$
X_v^{(j+1)}=\Phi_{0,\ell}\,Y_n
+\sum_{i=0}^{\ell-1}\Phi_{i+1,\ell}\Big[b_{n+i}\,v_{n+i}\big(X_{u_i}^{(j)}\big)+\eta_{u_{i+1}}\Big],
\qquad
\Phi_{i,\ell}=\prod_{k=i}^{\ell-1}a_{n+k},
$$

with the empty product equal to 1. The step from $\ell$ to $\ell+1$ multiplies every existing
term by $a_{n+\ell}$, which extends each $\Phi$ by one factor, and adds the new term with
$\Phi_{\ell+1,\ell+1}=1$. The linear part of the kernel is propagated exactly by $\Phi$, as in an
exponential integrator. Only the velocity is taken from the previous sweep.

In both forms, shared path prefixes are computed once, and all expensive target rows of a sweep
are evaluated in one batch.

## 5. Conditional proposal law and exactness

Let $\mathcal A_u$ contain everything fixed before the round (the committed state, the proposal's
memory, and the model conditioning) and the innovations on the path from $r$ to $u$. It excludes the
outgoing innovations $\lbrace\eta_v : v\in C(u)\rbrace$.

**Lemma.** If the update map is row-local, then for every sweep $j$ both $X_u^{(j)}$ and
$\widetilde\mu_u^{(j)}$ are $\mathcal A_u$-measurable.

*Proof.* Induction over depth, with the statement holding for all sweeps at once. The root has
$X_r^{(j)}=Y_n$, and $\widetilde\mu_r^{(j)}$ is a function of $Y_n$ and fixed quantities. Let
$u=\operatorname{pa}(v)$ and assume the claim for $u$. Then $X_v^{(j)}=\widetilde\mu_u^{(j)}+\eta_v$ is a
function of an $\mathcal A_u$-measurable quantity and of $\eta_v$, and both belong to
$\mathcal A_v\supseteq\mathcal A_u$. For $j\ge1$,
$\widetilde\mu_v^{(j)}=\mathcal M_{s(v)}\big(X_v^{(j-1)};X_v^{(j)}\big)$ depends on row $v$ alone, so it is
$\mathcal A_v$-measurable. For $j=0$, the draft mean $m^p_{s(v)}(X_v^{(0)})$ is a function of the
node state and of fixed proposal memory. $\square$

Row locality is what the lemma needs. An update that mixed rows could feed a descendant of $u$,
and therefore $u$'s outgoing innovations, into $\widetilde\mu_u$.

The outgoing innovations are independent of $\mathcal A_u$, so

$$
X_v^{(J)}\mid\mathcal A_u\ \overset{\mathrm{i.i.d.}}{\sim}\ \mathcal N\big(\widetilde\mu_u^{(J)},\,\sigma_{s(u)}^2I\big),
\qquad v\in C(u).
$$

This conditional Gaussian law is what RMC, D-GRS, and PAWS require. The marginal law of a node is
generally a Gaussian mixture when the drift is nonlinear. Verification at $u$ compares

$$
P_u^{(J)}=\mathcal N\big(\widetilde\mu_u^{(J)},\,\sigma_s^2I\big)
\qquad\text{with}\qquad
Q_u^{(J)}=\mathcal N\big(m^q_s(X_u^{(J)}),\,\sigma_s^2I\big),
$$

and both means are $\mathcal A_u$-measurable. Given $\mathcal A_u$, the verifier returns a state
distributed as $Q_u^{(J)}$ whatever $\widetilde\mu_u^{(J)}$ is. So refinement changes acceptance
rates, and never the law of committed states.

For RMC, define the normalized displacement

$$
\Delta_u=\frac{m^q_s\big(X_u^{(J)}\big)-\widetilde\mu_u^{(J)}}{\sigma_s}.
$$

Two isotropic Gaussians with equal covariance differ only along $\Delta_u$, where they are
$\mathcal N(0,1)$ and $\mathcal N(\lVert\Delta_u\rVert,1)$. Their densities cross at the midpoint, so
$\operatorname{TV}=\Phi(\lVert\Delta_u\rVert/2)-\Phi(-\lVert\Delta_u\rVert/2)$, and the maximal
coupling accepts with probability

$$
\Pr(\mathrm{accept}\mid\mathcal A_u)
=1-\operatorname{TV}\big(P_u^{(J)},Q_u^{(J)}\big)
=2\Phi\Big(-\frac{\lVert\Delta_u\rVert}{2}\Big).
$$

For $K>1$, D-GRS and PAWS use the same conditional kernels with their own multi-proposal rules.

## 6. Finite-depth propagation

**Proposition.** For any update map satisfying (C), in exact arithmetic,

$$
X_v^{(j)}=X_v^\star\quad\text{for every }|v|\le j,
$$

and every parent with $|u| < j$ has $\widetilde\mu_u^{(j)}=m^q_{s(u)}\big(X_u^{(j)}\big)$.

*Proof.* Outer induction on the sweep $j$. For $j=0$, only the root qualifies. Assume the claim
for $j$, and run an inner induction on depth within sweep $j+1$. The root is fixed. Take $v$
with $|v|\le j+1$ and $u=\operatorname{pa}(v)$, so $|u|\le j$. The inner hypothesis gives
$X_u^{(j+1)}=X_u^\star$, and the outer hypothesis gives $X_u^{(j)}=X_u^\star$. By (C),

$$
\widetilde\mu_u^{(j+1)}=\mathcal M_{s(u)}\big(X_u^\star;\,X_u^\star\big)=m^q_{s(u)}\big(X_u^\star\big),
\qquad
X_v^{(j+1)}=m^q_{s(u)}\big(X_u^\star\big)+\eta_v=X_v^\star .
$$

The same computation at $|u|\le j$ gives the statement about means. $\square$

No contraction assumption is needed: each sweep propagates exact target information at least one
level deeper. For a tree of depth $L$, $J\ge L$ gives $X^{(J)}=X^\star$. After $J$ sweeps, every
parent with $|u| < J$ verifies with $\Delta_u=0$ and is accepted by RMC with probability 1. The
draft is a no-op refiner: $X^{(0)}$ already satisfies its own recursion, so applying the
proposal's own drift again reproduces it.

## 7. Mismatch at verification

By (C), $m^q_s(x)=\mathcal M_s(x;x)$. The mismatch at a parent is therefore the change of the
frozen quantity over the last sweep, read at the final anchor:

$$
\sigma_s\Delta_u=\mathcal M_s\big(X_u^{(J)};\,X_u^{(J)}\big)-\mathcal M_s\big(X_u^{(J-1)};\,X_u^{(J)}\big).
$$

For a split update this is $N_s\big(X_u^{(J)}\big)-N_s\big(X_u^{(J-1)}\big)$; only the frozen part
contributes. Let $e_u=X_u^{(J)}-X_u^{(J-1)}$. For the churn kernels and the same iterates:

$$
\begin{aligned}
\sigma_s\Delta_u^{\mathrm{inc}}&=(a_s-1)\,e_u+b_s\big[v_s(X_u^{(J)})-v_s(X_u^{(J-1)})\big],\\
\sigma_s\Delta_u^{\mathrm{drift}}&=b_s\big[v_s(X_u^{(J)})-v_s(X_u^{(J-1)})\big].
\end{aligned}
$$

So, exactly and without linearization,

$$
\sigma_s\big(\Delta_u^{\mathrm{inc}}-\Delta_u^{\mathrm{drift}}\big)=(a_s-1)\,e_u .
$$

The whole-increment update reads the linear term $(a_s-1)x$ off the stale snapshot instead of
evaluating it at the final anchor. This changes neither the law ([§5](#5-conditional-proposal-law-and-exactness))
nor the fixed point ([§6](#6-finite-depth-propagation)). It only changes acceptance at parents
with $|u|\ge J$, where $e_u\neq0$.

## 8. Size of the stale affine term

Write $\hat x_0(x)=\mathbb E[x_0\mid x]$ for the interpolant $x=(1-t)x_0+t\xi$. The velocity is
$v=(x-\hat x_0)/t$, so its Jacobian is

$$
J_v=\frac{I-J_{\hat x_0}}{t}.
$$

The Jacobian of the posterior mean is a scaled posterior covariance. Differentiate
$\log p(x_0\mid x)=\log p(x_0)+\log\mathcal N\big(x;(1-t)x_0,t^2I\big)-\log p_t(x)$ in $x$. Tweedie's
formula, $\nabla\log p_t(x)=\big((1-t)\hat x_0-x\big)/t^2$, gives

$$
\nabla_x\log p(x_0\mid x)=\frac{(1-t)x_0-x}{t^2}-\nabla\log p_t(x)=\frac{1-t}{t^2}\big(x_0-\hat x_0\big),
$$

$$
J_{\hat x_0}=\int x_0\,\nabla_x p(x_0\mid x)^{\top}\,dx_0
=\frac{1-t}{t^2}\,\mathbb E\big[x_0(x_0-\hat x_0)^{\top}\mid x\big]
=\frac{1-t}{t^2}\operatorname{Cov}[x_0\mid x]\succeq0 .
$$

Linearize both mismatches in $e_u$ along an eigenvector of $J_{\hat x_0}$ with eigenvalue
$\lambda\ge0$:

$$
\sigma_s\Delta_u^{\mathrm{inc}}\approx-h\left[\frac{\varepsilon^2}{1-t}+\frac{(1+\varepsilon^2)(1-\lambda)}{t}\right]e_u,
\qquad
\sigma_s\Delta_u^{\mathrm{drift}}\approx-h\,\frac{(1+\varepsilon^2)(1-\lambda)}{t}\,e_u,
$$

$$
\frac{\lVert\Delta_u^{\mathrm{inc}}\rVert}{\lVert\Delta_u^{\mathrm{drift}}\rVert}
\approx\left\lvert\,1+\frac{\varepsilon^2\,t}{(1+\varepsilon^2)(1-t)(1-\lambda)}\,\right\rvert .
$$

There are three regimes:

- When $0\le\lambda < 1$, both coefficients are negative, so the terms add and the ratio
  exceeds 1. The ratio grows like $1/(1-t)$ toward the noise end. It also grows as $\lambda\to1$
  at low noise, where the velocity term shrinks.
- When $\lambda > 1$, the velocity coefficient changes sign, and the two terms partly cancel.
  This happens in directions where the posterior over $x_0$ is still split, for example between
  mixture components at high noise. There the stale term can reduce the mismatch, so the
  frozen-drift update is not better in every direction. It is better on average in every
  measured configuration ([§13](#13-measured-effect-on-the-gaussian-mixture)).
- When $\varepsilon=0$, $a_s=1$ and the updates coincide.

The size of the stale term alone, normalized by the transition std
$\sigma_s=\varepsilon\,g(t)\sqrt h$, is

$$
\frac{\lVert(a_s-1)\,e_u\rVert}{\sigma_s}
=\frac{h\varepsilon^2}{1-t}\cdot\frac{\lVert e_u\rVert}{\varepsilon\sqrt{2t/(1-t)}\,\sqrt h}
=\varepsilon\sqrt{\frac{h}{2t(1-t)}}\;\lVert e_u\rVert .
$$

On the uniform grid $t_k=1-kh$, the first stochastic step has $1-t=h$, so $a_s-1=-\varepsilon^2$
exactly.

## 9. Error propagation across sweeps

Let $E_v^{(j)}=X_v^{(j)}-X_v^\star$. Subtracting the exact recursion from a split update gives

$$
E_v^{(j+1)}=\big[A_s(X_u^{(j+1)})-A_s(X_u^\star)\big]+\big[N_s(X_u^{(j)})-N_s(X_u^\star)\big]
\approx A_s'\,E_u^{(j+1)}+N_s'\,E_u^{(j)},
$$

with Jacobians taken at the exact tree:

| update | $A_s'$ | $N_s'$ |
|---|---|---|
| whole increment | $I$ | $(a_s-1)I+b_sJ_v$ |
| frozen drift | $a_sI$ | $b_sJ_v$ |

The two recursions differ by $(a_s-1)\big(E_u^{(j)}-E_u^{(j+1)}\big)$, the stale term again, now
acting on state errors.

Unroll from $(v,j)$ toward the root. An $A'$ factor keeps the sweep index, and an $N'$ factor
lowers it by one. The root's error is zero, so a nonzero term must reach the draft, sweep 0, at
an ancestor $w$ with $|w|\ge1$. The step that reaches sweep 0 is an $N'$ factor. Each term is
therefore a product of $|v|-|w|$ Jacobians, exactly $j$ of them $N'$ factors with the last one
fixed. That gives $\binom{|v|-|w|-1}{j-1}$ orderings, and

$$
\lVert E_v^{(j)}\rVert\lesssim
\sum_{w}\binom{|v|-|w|-1}{j-1}\,\lVert N'\rVert^{\,j}\,\lVert A'\rVert^{\,|v|-|w|-j}\,\lVert E_w^{(0)}\rVert,
$$

summed over the ancestors $w$ of $v$ with $1\le|w|\le|v|-j$. The bound is empty for $|v|\le j$,
which recovers [§6](#6-finite-depth-propagation). The error after $j$ sweeps scales with the $j$-th
power of the frozen part's Jacobian. In the $\lambda < 1$ directions, removing $(a_s-1)I$ from $N'$
shrinks it. The frozen-drift update also damps the new-error path by
$\lVert A'\rVert=a_s < 1$; for $\varepsilon\le1$ on the uniform grid, $a_s\in[1-\varepsilon^2,1)$.

## 10. Floating point, target-mean reuse, and cost

The statements above hold in exact arithmetic. Three floating-point facts make them usable.

**Bitwise stability of the converged prefix.** Assume the target, the hooks, and the callback are
deterministic. Then a node with $|v|\le j$ has the same bits after every sweep $j'\ge j$. By
induction, its parent is stable from sweep $j-1$. The parent's snapshot row, target mean, frozen
quantity, and anchor are then bit-identical inputs to identical operations in every later sweep.

**Exact means at converged parents.** When the callback also supplies `exact_target_means`,
`refine_tree` evaluates the frozen-drift mean as

$$
\widetilde\mu_u^{(j+1)}
=m^q_s\big(X_u^{(j)}\big)
+\Big[\operatorname{apply}_s\big(D,\,X_u^{(j+1)}\big)-\operatorname{apply}_s\big(D,\,X_u^{(j)}\big)\Big],
\qquad
D=D_s\big(X_u^{(j)}\big).
$$

By the inverse property, this equals $\operatorname{apply}_s(D,X_u^{(j+1)})$ in exact arithmetic. When the
parent has not moved, the bracket is exactly zero. The proposal mean is then the cached target
mean bit for bit, so converged parents verify with $\Delta_u=0$ exactly. Without the anchoring,
$\operatorname{apply}(\operatorname{freeze}(y,m),y)$ would return $m$ only up to rounding. The
whole-increment update computes $X_u+(m-X_u)$. That equals $m$ bit for bit only when $m-X_u$ is
exact, which Sterbenz's lemma guarantees when $m$ and $X_u$ are within a factor of two.

**Exact final target-mean reuse.** The last sweep evaluates $m^q_s(X_u^{(J-1)})$, while
verification needs $m^q_s(X_u^{(J)})$. A cached row is reused only when all of the following
match:

- target instance
- image index
- logical node and flat-buffer identity
- step
- represented state, under exact elementwise equality

Then

$$
X_u^{(J)}=X_u^{(J-1)}\ \text{bitwise}
\ \Longrightarrow\
m^q_s\big(X_u^{(J)}\big)=m^q_s\big(X_u^{(J-1)}\big).
$$

No tolerance is used. A custom backend that does not implement exact row comparison
conservatively re-evaluates every row. GPU nondeterminism can only reduce reuse; it cannot cause
approximate states to be treated as identical.

**Target calls per round.** Internal nodes have $|u|\le L_n-1$. Bitwise stability makes every
internal node with $|u|\le J-1$ reusable. A final verification call is needed exactly when some
internal node has $|u|\ge J$, that is, when $J < L_n$. Without leaf evaluation, a scalar round
therefore costs

$$
J+\mathbf 1\lbrace J < L_n\rbrace
$$

target calls, excluding proposal-owned calls. A batched iteration costs

$$
J+\mathbf 1\lbrace \exists i : J < L_{n_i}\rbrace .
$$

With `evaluate_leaves=True`, leaves are not refinement parents, so a final call is still needed
whenever active leaves are requested. Reuse between refinement sweeps is intentionally not
implemented. It could reduce target batch volume after shallow nodes stabilize, but normally not
target-call depth while deeper nodes are still changing.

## 11. Callback contract

A callback receives one row for every internal node, potentially including ancestors and
descendants of one another. Output row $i$ must be local:

$$
\mathrm{row}_i^{(j)}=f_j\big(b_i,\,u_i,\,s_i,\,X_i^{(j)},\,\widetilde\mu_i^{(j)},\,\sigma_i\big).
$$

It may use fixed parameters and the conditioning for image $b_i$. It must not inspect, reduce
over, or mix other rows: cross-row dependence can make an ancestor's mean depend on its outgoing
noise and invalidate the conditional proposal law ([§5](#5-conditional-proposal-law-and-exactness)).

Callbacks always return a `RefinementUpdate` with exactly one of `increments` and `drifts`:

```python
RefinementUpdate(increments=increments, exact_target_means=target_means_or_none)
RefinementUpdate(drifts=drifts, exact_target_means=target_means_or_none)
```

`increments[i]` is added to the rebuilt parent. `drifts[i]` is passed to `target.apply_drift` at
the rebuilt parent and its step, so row locality then extends to `apply_drift`. With
`exact_target_means`, the drift mean is anchored as in [§10](#10-floating-point-target-mean-reuse-and-cost).
Both have the shape of `parent_states`.

`exact_target_means[i]`, when present, is a correctness-bearing assertion that it equals the
target mean at the request's exact image, step, and state. An incorrect assertion can invalidate
sampling. With `check_contract=True`, every reused assertion is checked in one unaccounted target
batch.

Target transitions, their hooks, and callbacks must be deterministic and row-independent for
fixed image conditioning, step, and state. Batch partitioning is an execution choice, not part of
the mathematical map.

## 12. Accounting

Each round separates target work into:

- `proposal_target_calls` and `proposal_target_states_evaluated`;
- `refinement_target_calls` and `refinement_target_states_evaluated`;
- `verification_target_calls` and `verification_target_states_evaluated`;
- `verification_target_means_reused`.

The three call categories and the three state-count categories sum to the round totals. Global
`SamplingResult.target_calls` remains the speedup denominator.

## 13. Measured effect on the Gaussian mixture

Setup: `experiments/gm/models.py` with $d=512$, five components, and $T=30$, which gives 28
speculative steps. The proposal is `DelayedDriftProposal` with `prefetch="nearest"`.

**The ratio of §8 on identical iterates.** $\varepsilon=0.6$, a chain of depth 6, $J\in\lbrace1,2\rbrace$.
Both mean formulas are applied to the same iterates, and each row is the median over parents with
$e_u\neq0$:

| $t$ | $a_s-1$ | measured ratio | formula, $\lambda=0$ |
|---|---|---|---|
| 0.897 | −0.120 | 1.98 | 3.30 (directions with $\lambda > 1$; 3 nodes) |
| 0.862 | −0.090 | 2.67 | 2.66 |
| 0.793 | −0.060 | 2.02 | 2.02 |
| 0.690 | −0.040 | 1.60 | 1.59 |
| 0.518 | −0.026 | 1.30 | 1.28 |
| 0.311 | −0.018 | 1.14 | 1.12 |
| 0.070 | −0.013 | 1.10 | 1.02 ($\lambda > 0$) |

At $\varepsilon=0.1$ the ratio stays at or below 1.06.

**End to end.** Each configuration uses 40 paired trajectories, with shared initial states and
sampler random streams. RMC runs on the chain matched to the $K=2$, $L=4$ verification budget,
which has depth 15. Values are means over trajectories:

| $\varepsilon$ | rule and tree | $J$ | acceptance, increment → drift | speedup, increment → drift |
|---|---|---|---|---|
| 0.6 | RMC, chain of depth 15 | 1 | 0.796 → 0.808 | 2.05 → 2.17 |
| 0.6 | RMC, chain of depth 15 | 2 | 0.880 → 0.895 | 2.08 → 2.27 |
| 0.6 | D-GRS, $K=2$, $L=4$ | 1 | 0.859 → 0.879 | 1.66 → 1.66 |
| 0.6 | PAWS, $K=2$, $L=4$ | 1 | 0.885 → 0.898 | 1.69 → 1.71 |
| 0.3 | RMC, chain of depth 15 | 1 | 0.822 → 0.833 | 2.35 → 2.40 |
| 0.3 | RMC, chain of depth 15 | 2 | 0.905 → 0.909 | 2.49 → 2.55 |
| 0.3 | D-GRS, $K=2$, $L=4$ | 1 | 0.904 → 0.909 | 1.71 → 1.71 |
| 0.3 | PAWS, $K=2$, $L=4$ | 1 | 0.930 → 0.934 | 1.77 → 1.78 |

With $K=2$, $L=4$, and $J=2$, both updates reach the same speedup, 1.27. There a round costs
$J+1=3$ target calls and commits at most four steps. To reproduce these numbers at scale, run
`experiments/gm/picard_sweep.py` once with `--picard-update increment` and once with the default,
into separate `--out` directories.
