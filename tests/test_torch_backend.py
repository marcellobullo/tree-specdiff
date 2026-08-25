"""Both couplings on the PyTorch backend. `python tests/test_torch_backend.py`.

The rest of the suite runs on NumPy, but two guards in the rules exist
*specifically* because of how this backend draws uniforms:

    TorchBackend.uniform is `torch.rand(())`, which is **float32**, so it lands
    on a 2^-24 grid and returns exactly 0.0 with probability ~6e-8 -- nine
    orders of magnitude more often than NumPy's 2^-53.

That is the reason RMC compares `math.log1p(-u)` rather than `math.log(u)`
(which raises at u = 0) and D-GRS compares `u < beta` rather than `u <= beta`
(which would accept where beta is exactly 0). Without this file the backend
those guards were written for is the one backend nothing exercises.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from specdiff import (  # noqa: E402
    DelayedDriftProposal,
    DraftTree,
    SpeculativeSampler,
    check_exactness,
    create_verifier,
)
from specdiff.kernels import ConstantSchedule, TargetTransition  # noqa: E402
from specdiff.ops import resolve_backend, standard_normal_sf  # noqa: E402
import specdiff.verifiers.rank1 as rank1  # noqa: E402
from specdiff.verifiers.rank1 import DEFAULT_DEGENERATE_TOL  # noqa: E402

DIM = 8


def _predicted_acceptance(delta, num_children):
    """Theorem 2 for D-GRS; collapses to eq. (16) at K = 1."""
    level, mass = 0.0, 1.0
    for _ in range(num_children):
        level += mass
        tau = math.log(level) / delta
        mass = standard_normal_sf(tau - delta / 2.0) - level * standard_normal_sf(
            tau + delta / 2.0
        )
    return 1.0 - mass


# --------------------------------------------------------------- the premise
def test_torch_uniform_is_float32_and_can_return_zero():
    """The fact both guards are built around.

    If this ever becomes float64, the guards stay correct but their urgency
    drops by nine orders of magnitude -- so it is worth failing loudly if the
    assumption changes rather than leaving the comments stale.
    """
    ops = resolve_backend(torch.zeros(DIM))
    gen = torch.Generator().manual_seed(0)
    values = [ops.uniform(gen) for _ in range(2000)]
    assert all(0.0 <= v < 1.0 for v in values)

    assert torch.rand((), generator=torch.Generator().manual_seed(0)).dtype is torch.float32
    # 2^-24 granularity: every draw is an integer multiple of it.
    grid = 2.0**-24
    assert all(abs(v / grid - round(v / grid)) < 1e-6 for v in values)


# ------------------------------------------------------------------ exactness
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rmc_is_exact_on_torch(dtype):
    for delta in (0.0, 1.0, 3.0):
        report = check_exactness(
            create_verifier("rmc"),
            delta=delta,
            num_children=1,
            seed=0,
            alpha=0.001,
            array_like=torch.zeros(DIM, dtype=dtype),
        )
        assert report.passed, report


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_children", [1, 2, 4])
def test_dgrs_is_exact_on_torch(dtype, num_children):
    for delta in (0.0, 1.0, 3.0):
        report = check_exactness(
            create_verifier("d-grs"),
            delta=delta,
            num_children=num_children,
            seed=0,
            alpha=0.001,
            array_like=torch.zeros(DIM, dtype=dtype),
        )
        assert report.passed, report


# ----------------------------------------------------------------- acceptance
def test_acceptance_rates_hold_on_torch():
    """Exactness alone would pass for a rule that never accepts; this is what
    ties each rule to its own algorithm, on this backend too."""
    n = 20000
    cases = [("rmc", 1), ("d-grs", 1), ("d-grs", 2), ("d-grs", 4)]
    for name, num_children in cases:
        report = check_exactness(
            create_verifier(name),
            delta=1.0,
            num_children=num_children,
            seed=1,
            num_samples=n,
            array_like=torch.zeros(DIM),
        )
        predicted = (
            2.0 * standard_normal_sf(0.5)
            if name == "rmc"
            else _predicted_acceptance(1.0, num_children)
        )
        stderr = math.sqrt(predicted * (1.0 - predicted) / n)
        assert abs(report.acceptance_rate - predicted) < 4.0 * stderr, (
            f"{name} K={num_children}: measured {report.acceptance_rate:.4f}, "
            f"predicted {predicted:.4f}"
        )


# ------------------------------------------------------- half precision
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reductions_are_range_safe_in_half_precision(dtype):
    """`norm` and `dot` must reduce in at least float32.

    float16 has a 5-bit exponent, so squaring inside the reduction underflows
    to 0 below ~2.4e-4 and overflows to inf above ~256 -- both ordinary state
    magnitudes. `norm` feeds `Rank1Frame.delta`, so an underflow there does not
    perturb the coupling, it *disables* it: delta reads as exactly 0, every
    frame looks degenerate, and every rule accepts unconditionally.
    """
    ops = resolve_backend(torch.zeros(DIM, dtype=dtype))
    for magnitude in (1e-5, 1e-3, 1.0, 300.0):
        v = torch.zeros(DIM, dtype=dtype)
        v[0] = magnitude
        got = ops.norm(v)
        assert math.isfinite(got), f"norm overflowed at {magnitude}"
        assert got > 0.0, f"norm underflowed to zero at {magnitude}"
        assert abs(got - magnitude) / magnitude < 0.01
        assert ops.dot(v, v) > 0.0, f"dot underflowed at {magnitude}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name,num_children", [("rmc", 1), ("d-grs", 4)])
def test_rules_are_exact_in_half_precision(dtype, name, num_children):
    """The assumption the degenerate shortcut rests on, made executable.

    The shortcut accepts unconditionally, which costs `TV = delta / sqrt(2 pi)`
    of exactness each time it fires. That is justified only while the tolerance
    is small. When the tolerance was `sqrt(eps)` of the state dtype it was
    3.1e-2 in float16 and 8.8e-2 in bfloat16 -- and `check_exactness` detects
    those as non-exact. Nothing tested half precision, so nothing caught it.
    """
    for delta in (0.0, 0.02, 1.0):
        report = check_exactness(
            create_verifier(name),
            delta=delta,
            num_children=num_children,
            seed=0,
            alpha=0.001,
            array_like=torch.zeros(DIM, dtype=dtype),
        )
        assert report.passed, report


def test_old_dtype_derived_tolerance_is_detectably_biased():
    """The regression guard, exercising the real rules rather than the reasoning.

    `DEFAULT_DEGENERATE_TOL` used to be `sqrt(eps)` of the state dtype, which is
    0.031 in float16. Inside that window the coupling skipped its acceptance
    test and accepted unconditionally, costing `TV = delta / sqrt(2 pi)`.

    Restoring that value here must make `check_exactness` fail. If it ever
    stops failing, either the tolerance moved back or the exactness test lost
    its power -- and the current constant is no longer justified by anything.
    """
    old_float16_tol = math.sqrt(float(torch.finfo(torch.float16).eps))
    assert DEFAULT_DEGENERATE_TOL < old_float16_tol / 1000

    delta = 0.02  # inside the old window, outside the new one
    previous = rank1.DEGENERATE_TOL
    try:
        rank1.DEGENERATE_TOL = old_float16_tol
        biased = check_exactness(
            create_verifier("d-grs"), delta=delta, num_children=4, seed=0,
            alpha=0.001, num_samples=50000,
            array_like=torch.zeros(DIM, dtype=torch.float16),
        )
    finally:
        rank1.DEGENERATE_TOL = previous

    assert not biased.passed, (
        f"the old dtype-derived tolerance is no longer detectable ({biased}); "
        "re-derive DEFAULT_DEGENERATE_TOL rather than trusting it"
    )

    # ... and with the tolerance as shipped, the same configuration is exact.
    fixed = check_exactness(
        create_verifier("d-grs"), delta=delta, num_children=4, seed=0,
        alpha=0.001, num_samples=50000,
        array_like=torch.zeros(DIM, dtype=torch.float16),
    )
    assert fixed.passed, fixed



# ------------------------------------------------------------------ end to end
class _ShiftKernel(TargetTransition):
    def means(self, indices_in_batch, states, steps):
        return states + 0.1


@pytest.mark.parametrize(
    "name,tree",
    [("rmc", DraftTree.chain(3)), ("d-grs", DraftTree.uniform(branching=3, lookahead=2))],
)
def test_runs_end_to_end_on_torch(name, tree):
    target = _ShiftKernel()
    sampler = SpeculativeSampler(
        target=target,
        proposal=DelayedDriftProposal(target),
        schedule=ConstantSchedule(0.5),
        tree=tree,
        verifier=create_verifier(name),
        num_steps=20,
        check_contract=True,
    )
    result = sampler.sample(
        torch.zeros(DIM), rng=torch.Generator().manual_seed(0)
    )
    assert tuple(result.trajectory.shape) == (21, DIM)
    assert bool(torch.isfinite(result.trajectory).all())
    assert result.target_calls <= 20


if __name__ == "__main__":
    import itertools

    for fn, args in itertools.chain(
        [(test_torch_uniform_is_float32_and_can_return_zero, ())],
        [(test_rmc_is_exact_on_torch, (dt,)) for dt in (torch.float32, torch.float64)],
        [
            (test_dgrs_is_exact_on_torch, (dt, k))
            for dt in (torch.float32, torch.float64)
            for k in (1, 2, 4)
        ],
        [(test_acceptance_rates_hold_on_torch, ())],
        [(test_reductions_are_range_safe_in_half_precision, (dt,))
         for dt in (torch.float16, torch.bfloat16)],
        [
            (test_rules_are_exact_in_half_precision, (dt, n, k))
            for dt in (torch.float16, torch.bfloat16)
            for n, k in (("rmc", 1), ("d-grs", 4))
        ],
        [(test_old_dtype_derived_tolerance_is_detectably_biased, ())],
        [
            (test_runs_end_to_end_on_torch, (n, t))
            for n, t in (
                ("rmc", DraftTree.chain(3)),
                ("d-grs", DraftTree.uniform(branching=3, lookahead=2)),
            )
        ],
    ):
        fn(*args)
        print(f"ok  {fn.__name__}{args if args else ''}")
    print("\nall passed")


# --------------------------------------------------------------- device placement
def _non_cpu_device():
    """A real accelerator to test against, or None.

    cuda on a server, mps on a Mac. The bug this guards against is invisible on
    cpu, which is why the rest of this file never caught it: `torch.rand`
    allocates on the *default* device and refuses a generator from anywhere
    else, so a run on cuda died at its first acceptance test.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_non_cpu_device() is None, reason="no accelerator available")
def test_uniform_accepts_a_generator_from_the_states_device():
    dev = _non_cpu_device()
    ref = torch.zeros(3, device=dev)
    ops = resolve_backend(ref)
    rng = ops.make_rng(0, ref=ref)
    assert str(rng.device).startswith(dev)

    u = ops.uniform(rng)                      # must not raise
    assert 0.0 <= u < 1.0
    # Still a float32 draw, which is what the guards in RMC and D-GRS assume.
    assert isinstance(u, float)


@pytest.mark.skipif(_non_cpu_device() is None, reason="no accelerator available")
def test_a_whole_sample_runs_on_the_accelerator():
    """End to end off cpu: the failure above only surfaced under a real run."""
    dev = _non_cpu_device()
    target = _ShiftKernel()
    sampler = SpeculativeSampler(
        target=target,
        proposal=DelayedDriftProposal(target),
        schedule=ConstantSchedule(0.3),
        tree=DraftTree.uniform(branching=2, lookahead=2),
        verifier=create_verifier("d-grs"),
        num_steps=8,
    )
    init = torch.zeros(4, device=dev)
    ops = resolve_backend(init)
    result = sampler.sample(init, rng=ops.make_rng(0, ref=init))
    assert str(result.sample.device).startswith(dev)
    assert torch.isfinite(result.sample).all()
