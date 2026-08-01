# specdiff documentation

Start with the [project README](../README.md) for what this is, the notation table, and a
five-line quick start. These four documents go deeper.

| document | read it if you want to |
| --- | --- |
| [writing-a-verifier.md](writing-a-verifier.md) | **implement a verification rule** — the contract, rank-1 coordinates, how to test for exactness, and the traps. This is the library's primary extension point and the guide most readers want |
| [models.md](models.md) | sample from your own diffusion model: the target, the schedule, proposals, choosing a draft tree, dtype rules, adding an array backend |
| [architecture.md](architecture.md) | understand or modify the driver: the three phases of a round, step indexing, horizon truncation, cost accounting, and why the seams are where they are |
| [api-reference.md](api-reference.md) | look up a signature or an attribute |

## Common starting points

**"I want to implement Algorithm 1 or 2."**
[writing-a-verifier.md](writing-a-verifier.md), then
[the section on the two stubs](writing-a-verifier.md#implementing-the-papers-two-algorithms).

**"I want to know if speculation is worth it on my model."**
[Measuring your headroom first](writing-a-verifier.md#measuring-your-headroom-first) — a probe
that is exact by construction, so it is safe against a production model, and reports the
speedup ceiling before you write any coupling.

**"My rule passes `check_contract` but samples look wrong."**
That is exactly what [`check_exactness`](writing-a-verifier.md#testing-for-exactness) is for.
`check_contract` only covers the checkable half of the contract; exactness needs the
statistical test.

**"Where did the number in `result.summary()` come from?"**
[Cost accounting](architecture.md#cost-accounting), and
[Results](api-reference.md#results) for the field list.

**"Why is the batched speedup lower than the per-trajectory speedup?"**
It is supposed to be — one target call serves every live trajectory, so the batch advances at
the pace of its slowest member. [The batched driver](architecture.md#the-batched-driver).
