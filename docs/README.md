# specdiff documentation

Begin with the [project README](../README.md) for an overview, notation, and quick start. The
documents below cover implementation and extension details.

| document | read it if you want to |
| --- | --- |
| [writing-a-verifier.md](writing-a-verifier.md) | implement a verification rule: contract, rank-1 coordinates, exactness tests, and common implementation errors |
| [models.md](models.md) | sample from your own diffusion model: the target, the schedule, proposals, choosing a draft tree, dtype rules, adding an array backend |
| [architecture.md](architecture.md) | understand or modify the sampler: round phases, step indexing, horizon truncation, cost accounting, and component boundaries |
| [api-reference.md](api-reference.md) | look up a signature or an attribute |

## Common starting points

**"I want to write a coupling of my own."**
[writing-a-verifier.md](writing-a-verifier.md), then
[the section on the paper's two rules](writing-a-verifier.md#the-papers-two-algorithms).
Both are implemented — `rmc` is the short one to read first, `d-grs` is the one that shows what
a sequence coupling over `K` proposals costs.

**"I want to know if speculation is worth it on my model."**
[Measuring your headroom first](writing-a-verifier.md#measuring-your-headroom-first) — a probe
that is exact by construction, so it is safe against a production model, and reports the
speedup ceiling before you write any coupling.

**"My rule passes `check_contract` but samples look wrong."**
Use [`check_exactness`](writing-a-verifier.md#testing-for-exactness). `check_contract` covers
structural invariants, while distributional exactness requires a statistical test.

**"Where did the number in `result.summary()` come from?"**
[Cost accounting](architecture.md#cost-accounting), and
[Results](api-reference.md#results) for the field list.

**"Why is the batched speedup lower than the per-trajectory speedup?"**
One target call serves every active trajectory, so the batch advances at the pace of its
slowest member. See [The batched sampler](architecture.md#the-batched-sampler).
