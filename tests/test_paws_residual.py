"""Closed-form residual CDF: exactness, conditioning, and sweep regressions.

The residual needs no quadrature, so quadrature appears here only as an oracle,
and two different oracles are needed because neither covers the whole range.

* Split quadrature checks the *formula*. It must be split at the crossings: a
  crossing is an interior kink and QAGS is not reliable across one. It is also
  blind to a span whose mass underflows float64 intermediates.
* Arbitrary precision checks the *arithmetic*, by evaluating the same closed
  form at 50 digits. It reaches the spans quadrature cannot, and it is what
  shows the small-gap forms doing their job.
"""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import kstest

import specdiff.verifiers.paws as paws
from specdiff.ops import NumpyBackend
from specdiff.verifiers.paws import _ResidualCDF, _SelectedDensity, _shift_deficit

# Exact gaps/weights from rejected transitions in the six reported eps=0.1 cells.
REPORTED = [
    (12.12573909835101, (0., 1.)),
    (1.8194288234109475, (0., 1.)),
    (1.8201586920349668, (0., 1.)),
    (2.4614274614193064, (0., 0., 1.)),
    (2.6582725430907197, (0., 0., 1.)),
    (2.461720818337694, (0., 0., 1.)),
    (3.3271155672828687, (5.502742639047795e-7, 0., 0., 0., .9999994497257361)),
    (3.099174247650213, (4.3276553759912984e-6, 0., 0., 0., .999995672344624)),
    (12.279826303654824, (0., 0., 0., 0., 1.)),
]
POLICIES = ("optimized", "uniform", "max")
NEGLIGIBLE = 1e-100
"""A span below this carries no weight in any sampling decision; float64
quadrature cannot resolve it, so it is checked in arbitrary precision instead."""


def crossings(residual, points):
    """Every root of ``log f - log g``, found the way ``_spans`` does.

    Sign changes are detected on booleans rather than on a product, so a grid
    point that lands exactly on a root is not silently skipped.
    """
    grid = np.linspace(*residual.bounds, points)
    excess = np.array([residual.log_excess(float(s)) for s in grid])
    roots = []
    for i in np.nonzero((excess[:-1] > 0) != (excess[1:] > 0))[0]:
        left, right = float(grid[i]), float(grid[i + 1])
        roots.append(left if excess[i] == 0 else right if excess[i + 1] == 0 else
                     brentq(residual.log_excess, left, right, xtol=1e-15, rtol=8.9e-16))
    return roots


def quadrature_mass(residual, a, b):
    """Residual mass on a span by quadrature split at every crossing, plus
    whether SciPy actually converged.

    The status is returned rather than discarded. Splitting at the crossings
    removes the interior kink, but the oracle still fails where the float
    density is a difference of tiny numbers, and a warned result must not be
    used to judge the closed form.
    """
    def density(s):
        return max(0., math.exp(-.5 * (s - residual.delta) ** 2 - paws._LOG_SQRT_2PI)
                   - math.exp(-.5 * s * s - paws._LOG_SQRT_2PI
                              + residual.selected.log_beta(s)))
    lo = a if math.isfinite(a) else residual.bounds[0]
    hi = b if math.isfinite(b) else residual.bounds[1]
    inner = [c for c in crossings(residual, 4001) if lo < c < hi]
    out = quad(density, lo, hi, points=inner or None, limit=400,
               epsabs=1e-300, epsrel=1e-13, full_output=True)
    return out[0], len(out) == 3


def exact_mass(residual, a, b, digits=50):
    """The same closed form at `digits` precision, where nothing cancels."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = digits
    weights, k = residual.selected.weights, len(residual.selected.weights)
    delta = mp.mpf(repr(residual.delta))

    def selected_cdf(s):
        if s == -mp.inf:
            return mp.mpf(0)
        if s == mp.inf:
            return mp.mpf(1)
        u = mp.ncdf(s)
        return mp.fsum(
            mp.fsum(weights[:j]) * mp.binomial(k, j) * u ** j * (1 - u) ** (k - j)
            for j in range(1, k + 1)
        )

    def target_cdf(s):
        return mp.mpf(0) if s == -mp.inf else mp.mpf(1) if s == mp.inf else mp.ncdf(s - delta)

    edges = [-mp.inf if not math.isfinite(x) and x < 0 else
             mp.inf if not math.isfinite(x) else mp.mpf(repr(x)) for x in (a, b)]
    return ((target_cdf(edges[1]) - target_cdf(edges[0]))
            - (selected_cdf(edges[1]) - selected_cdf(edges[0])))


# --- the reported failures ------------------------------------------------

@pytest.mark.parametrize("delta,weights", REPORTED)
def test_reported_sweep_cases_agree_with_split_quadrature(delta, weights):
    """These gaps made the old adaptive quadrature report non-convergence.

    Only the dominant spans are checked this way. On the rest the oracle either
    does not converge or underflows, and arbitrary precision takes over -- which
    is the whole reason the closed form replaced it.
    """
    residual = _ResidualCDF(delta, weights)
    assert residual.total > 0
    checked = 0
    for a, b in residual.spans:
        mass = residual.mass(a, b)
        assert mass >= 0
        reference, converged = quadrature_mass(residual, a, b)
        if converged and mass > 1e-6 * residual.total:
            assert mass == pytest.approx(reference, rel=1e-10)
            checked += 1
    assert checked, "no span was well enough scaled for the quadrature oracle"


@pytest.mark.parametrize("delta,weights", REPORTED)
def test_reported_sweep_cases_agree_with_arbitrary_precision(delta, weights):
    residual = _ResidualCDF(delta, weights)
    for a, b in residual.spans:
        exact = exact_mass(residual, a, b)
        assert residual.mass(a, b) == pytest.approx(float(exact), rel=1e-11)


def test_far_tail_span_survives_where_float_quadrature_underflows():
    """One reported case puts a span near s = -29, where every float64
    intermediate underflows. The closed form still returns its mass; the
    quadrature it replaced returns zero."""
    residual = _ResidualCDF(*REPORTED[0])
    tiny = [(a, b) for a, b in residual.spans if residual.mass(a, b) < NEGLIGIBLE]
    assert len(tiny) == 1
    a, b = tiny[0]
    assert quadrature_mass(residual, a, b)[0] == 0.0
    assert residual.mass(a, b) == pytest.approx(float(exact_mass(residual, a, b)), rel=1e-11)
    assert 0 < residual.mass(a, b) < NEGLIGIBLE


@pytest.mark.parametrize("delta,weights", REPORTED[:4])
def test_reported_sweep_case_quantiles_invert_the_closed_form(delta, weights):
    residual = _ResidualCDF(delta, weights)
    for u in (.001, .1, .5, .9, .999):
        s = residual.sample(u)
        assert math.isfinite(s)
        below = math.fsum(residual.mass(a, min(b, s)) for a, b in residual.spans if a < s)
        assert below / residual.total == pytest.approx(u, abs=1e-11)


# --- conditioning ---------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("delta", [1e-9, 1e-7, 3e-6, 1e-4, .01, .7, 3., 12., 200.])
def test_quantiles_are_monotone_and_finite(k, policy, delta):
    """Inverting a CDF requires the CDF to be monotone. The quadrature version
    was not: integrating to a moving endpoint fluctuated across the kink by far
    more than its own reported error, which produced spurious roots and
    quantiles outside the residual's support."""
    residual = _ResidualCDF(delta, paws.rank_weights(delta, k, policy))
    quantiles = [residual.sample(u) for u in
                 (0., 1e-12, 1e-6, .01, .3, .5, .7, .99, 1 - 1e-9, 1 - 1e-15, 1.)]
    assert all(map(math.isfinite, quantiles))
    assert quantiles == sorted(quantiles)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7])
def test_former_crash_window_is_resolved(k):
    """For 1e-7 <= delta <= 3e-5 the optimizer's weights are uniform perturbed
    at order delta, so the integrand near a crossing was float noise and the
    retry ladder exhausted its budget and raised."""
    for delta in np.geomspace(1e-7, 1e-4, 24):
        residual = _ResidualCDF(float(delta), paws.rank_weights(float(delta), k))
        assert residual.total > 0
        assert math.isfinite(residual.sample(.5))


@pytest.mark.parametrize("k,policy", [(k, p) for k in (2, 4, 6) for p in POLICIES])
@pytest.mark.parametrize("delta", [1e-6, .05, .9, 4.])
def test_coarse_scan_finds_every_crossing(k, policy, delta):
    """Production scans at _SCAN_POINTS; a 40x finer scan must find no more."""
    residual = _ResidualCDF(delta, paws.rank_weights(delta, k, policy))
    fine = sorted(crossings(residual, 20001))
    found = sorted(c for span in residual.spans for c in span if math.isfinite(c))
    assert len(found) == len(fine)
    for got, want in zip(found, fine):
        assert got == pytest.approx(want, abs=1e-12)


# --- the two small-gap forms ----------------------------------------------

@pytest.mark.parametrize("delta", [1e-12, 1e-8, 1e-4, .02, .06, .5, 3.])
@pytest.mark.parametrize("x", [-6., -4., -.3, 0., .3, 4., 6.])
def test_shift_deficit_stays_relative_at_every_gap(x, delta):
    """Phi(x) - Phi(x - delta) by differencing loses every digit as delta -> 0,
    and the residual mass is built from exactly that difference."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    exact = mp.ncdf(mp.mpf(repr(x))) - mp.ncdf(mp.mpf(repr(x)) - mp.mpf(repr(delta)))
    assert _shift_deficit(x, delta) == pytest.approx(float(exact), rel=1e-12)
    assert _shift_deficit(math.inf, delta) == 0.
    assert _shift_deficit(-math.inf, delta) == 0.


def test_shift_deficit_beats_naive_cdf_differencing_at_a_tiny_gap():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    x, delta = 0.4, 1e-12
    exact = mp.ncdf(mp.mpf(repr(x))) - mp.ncdf(mp.mpf(repr(x)) - mp.mpf(repr(delta)))
    naive = float(ndtr(x) - ndtr(x - delta))
    error = lambda got: abs(got - float(exact)) / float(exact)
    assert error(naive) > 1e-5                       # every digit but the first few
    assert error(_shift_deficit(x, delta)) < 1e-14


@pytest.mark.parametrize("k", [1, 2, 4, 7])
def test_uniform_weights_give_exactly_the_normal_cdf(k):
    """Uniform ranks select the plain proposal, so the deviation is exactly 0 --
    not 0 to rounding. Every coefficient Lambda_j - j/K vanishes identically."""
    selected = _SelectedDensity((1.0 / k,) * k)
    assert all(selected.cdf_deviation(s) == 0.0 for s in (-9., -1., 0., .4, 3., 40.))


@pytest.mark.parametrize("weights", [(0., 1.), (.3, .1, .2, .4), (0., 0., 0., 0., 1.),
                                     (.5, 0., 0., .5), (1.,)])
def test_selected_cdf_matches_the_integrated_selected_density(weights):
    selected = _SelectedDensity(weights)
    cdf = lambda s: float(ndtr(s)) + selected.cdf_deviation(s)
    density = lambda s: math.exp(-.5 * s * s - paws._LOG_SQRT_2PI + selected.log_beta(s))
    for s in (-3., -.7, 0., 1.1, 4.):
        assert cdf(s) == pytest.approx(
            quad(density, -40., s, limit=400, epsabs=1e-16, epsrel=1e-13)[0], abs=1e-13)
    assert cdf(45.) == pytest.approx(1.0, abs=1e-15)
    assert selected.cdf_deviation(math.inf) == 0.0


# --- structure ------------------------------------------------------------

def test_disconnected_support_keeps_both_tails():
    residual = _ResidualCDF(.7, (0., 0., 0., 1.))
    assert len(residual.spans) == 2
    assert residual.pdf(-4.) > 0 and residual.pdf(0.) == 0 and residual.pdf(4.) > 0
    assert math.fsum(residual.masses) == pytest.approx(residual.total, rel=1e-15)
    assert all(m > 0 for m in residual.masses)
    assert residual.sample(.01) < 0 < residual.sample(.99)


def test_masses_are_additive_over_a_split_span():
    residual = _ResidualCDF(2.3, paws.rank_weights(2.3, 4))
    for a, b in residual.spans:
        lo = a if math.isfinite(a) else residual.bounds[0]
        hi = b if math.isfinite(b) else residual.bounds[1]
        cuts = np.linspace(lo, hi, 7)
        parts = math.fsum(residual.mass(float(x), float(y))
                          for x, y in zip(cuts[:-1], cuts[1:]))
        assert parts == pytest.approx(residual.mass(a, b), rel=1e-12)


def test_degenerate_residual_is_reported_not_silently_normalized():
    with pytest.raises(FloatingPointError, match="below numerical resolution"):
        _ResidualCDF(0.0, (.5, .5))


# --- the rejection implementation of the same residual --------------------

def closed_form_cdf(residual, s):
    """``P(S <= s)`` under the residual, from the closed form, as an oracle."""
    return math.fsum(
        residual.mass(a, min(max(s, a), b)) for a, b in residual.spans if a < s
    ) / residual.total


@pytest.mark.parametrize("delta,weights", [
    (.7, (0., 0., 0., 1.)),     # disconnected support, one span in each tail
    (1., (0., 1.)),             # the paper's worked K=2 example
    (2.5, (.3, .1, .2, .4)),
    (.05, (.25,) * 4),          # residual mass ~ delta / sqrt(2 pi); many trials
])
def test_rejection_loop_samples_the_closed_form_residual(delta, weights):
    """The two implementations are the same distribution, not merely both exact.

    Rejection never forms the support, so this is the check that a proposal
    landing in a gap between spans is discarded rather than corrected into one.
    """
    residual = _ResidualCDF(delta, weights)
    ops, rng = NumpyBackend(), np.random.default_rng(101)
    draws = np.array([paws._sample_residual_by_rejection(delta, weights, ops, rng, 10_000)[0]
                      for _ in range(20_000)])
    assert kstest(draws, np.vectorize(lambda s: closed_form_cdf(residual, s))).pvalue > 1e-3
    for a, b in residual.spans:
        assert np.any((draws > a) & (draws < b))
    gaps = zip([b for _, b in residual.spans[:-1]], [a for a, _ in residual.spans[1:]])
    assert not any(np.any((draws > b) & (draws < a)) for b, a in gaps)


@pytest.mark.parametrize("delta,weights", [(.7, (0., 0., 0., 1.)), (2.5, (.3, .1, .2, .4))])
def test_rejection_trial_count_is_the_reciprocal_residual_mass(delta, weights):
    """The cost the paper quotes: ``1 / eta`` trials per correction."""
    residual = _ResidualCDF(delta, weights)
    ops, rng = NumpyBackend(), np.random.default_rng(3)
    trials = [paws._sample_residual_by_rejection(delta, weights, ops, rng, 10_000)[1]
              for _ in range(4000)]
    assert np.mean(trials) == pytest.approx(1 / residual.total, rel=.08)
    assert min(trials) == 1


def test_trial_cap_defers_to_the_closed_form_instead_of_looping():
    """A cap of one trial makes the fallback the usual path, not the rare one.

    The fallback draw is an independent exact residual sample, so capping bounds
    the work without touching the law -- which is what this asserts, by running
    the same distributional check through a cap that fires almost every time.
    """
    delta, weights = .05, (.25,) * 4   # eta ~ 2%, so one trial almost never lands
    residual = _ResidualCDF(delta, weights)
    ops, rng = NumpyBackend(), np.random.default_rng(5)
    draws, spent = [], []
    for _ in range(6000):
        s, trials = paws._sample_residual_by_rejection(delta, weights, ops, rng, 1)
        draws.append(s)
        spent.append(trials)
    assert np.mean(spent) == 1.0
    assert kstest(np.array(draws),
                  np.vectorize(lambda s: closed_form_cdf(residual, s))).pvalue > 1e-3
