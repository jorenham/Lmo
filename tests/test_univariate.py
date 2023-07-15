# type: ignore

import functools

from pytest import approx
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np

import lmo

_R_MAX = 8
_T_MAX = 2
_N_MIN = _R_MAX + 2 * _T_MAX

st_r = st.integers(1, _R_MAX)
st_k = st.integers(2, _R_MAX)
st_t = st.integers(0, _T_MAX)
st_n = st.integers(_N_MIN, 50)
st_trim = st.tuples(st_t, st_t)
st_dtype = hnp.floating_dtypes(sizes=(64,))

__st_a_kwargs = {
    'dtype': st_dtype,
    'elements': st.floats(-1e4, -1e-2) | st.floats(1e-2, 1e4),
}
st_a1 = hnp.arrays(shape=st_n, **__st_a_kwargs)
st_a1_unique = hnp.arrays(shape=st_n, unique=True, **__st_a_kwargs)

st_a2 = hnp.arrays(
    shape=st.tuples(st_n, st.integers(1, 10)),
    **__st_a_kwargs
)

def cauchy_cdf(x: float) -> float:
    return np.arctan(x) / np.pi + 1 / 2

def cauchy_ppf(p: float) -> float:
    return np.tan(np.pi * (p - 1 / 2))

def expon_cdf(x: float, a: float = 1) -> float:
    return 1 - np.exp(-x / a)

def expon_ppf(p: float, a: float = 1) -> float:
    return -a * np.log(1 - p)


@given(a=st_a1, trim=st_trim)
def test_l_moment_zero(a, trim):
    l0 = lmo.l_moment(a, 0, trim)

    assert np.isscalar(l0)
    assert l0 == 1


@given(a=st_a1, r=st_r, trim=st_trim, w_const=st.floats(0.1, 10))
def test_l_moment_aweights_const(a, r, trim, w_const):
    l_r = lmo.l_moment(a, r, trim)

    w = np.full_like(a, w_const)
    l_r_w = lmo.l_moment(a, r, trim, aweights=w)

    assert np.isfinite(l_r_w)
    assert np.allclose(l_r_w, l_r)



@given(a=st_a1, r=st_r, trim=st_trim)
def test_l_ratio_unit(a, r, trim):
    tau = lmo.l_ratio(a, r, r, trim)

    assert np.allclose(tau, 1)


@given(a=st_a1 | st_a2)
def test_l_loc_mean(a):
    loc = a.mean(dtype=np.float_)
    l_loc = lmo.l_loc(a)

    assert l_loc.shape == loc.shape
    assert np.allclose(l_loc, loc, rtol=1e-4)


@given(a=st_a2)
def test_l_loc_mean_2d(a):
    locs = a.mean(axis=0, dtype=np.float_)
    l_locs = lmo.l_loc(a, axis=0)

    assert len(l_locs) == a.shape[1]
    assert l_locs.shape == locs.shape
    assert np.allclose(l_locs, locs, rtol=1e-4)

    l_locs_t = lmo.l_loc(a.T, axis=1)

    assert l_locs_t.shape == l_locs.shape
    assert np.allclose(l_locs_t, l_locs)


@given(x0=st.floats(-1e6, 1e6), n=st_n, dtype=st_dtype, trim=st_trim)
def test_l_loc_const(x0, n, dtype, trim):
    x = np.full(n, x0, dtype=dtype)
    l_1 = lmo.l_loc(x, trim)

    assert l_1 == approx(x0)


@given(
    x=st_a1 | st_a2,
    trim=st_trim,
    dloc=st.floats(-1e3, 1e3),
    dscale=st.floats(1e-3, 1e3)
)
def test_l_loc_linearity(x, trim, dloc, dscale):
    l1 = lmo.l_loc(x, trim)
    assert np.isfinite(l1)
    assert np.isscalar(l1)

    l1_add = lmo.l_loc(x + dloc, trim)
    assert l1_add == approx(l1 + dloc)

    l1_mul = lmo.l_loc(x * dscale, trim)
    assert l1_mul == approx(l1 * dscale)


@given(a=st_a1)
def test_l_scale_equiv_md(a):
    # half gini/abs mean difference (MD)
    n = len(a)
    scale = abs(a - a[:, None]).mean() / (2 - 2 / n)

    l2 = lmo.l_scale(a)

    assert l2.shape == scale.shape
    assert l2 == approx(scale)


@given(x0=st.floats(-1e6, 1e6), n=st_n, dtype=st_dtype, trim=st_trim)
def test_t_scale_const(x0, n, dtype, trim):
    x = np.full(n, x0, dtype=dtype)
    l2 = lmo.l_scale(x, trim)

    assert round(l2, 8) == 0


@given(x=st_a1 | st_a2, trim=st_trim, dloc=st.floats(-1e3, 1e3))
def test_l_scale_invariant_loc(x, trim, dloc):
    l2 = lmo.l_scale(x, trim)
    assert np.isfinite(l2)
    assert np.isscalar(l2)
    assert round(l2, 8) >= 0

    l2_add = lmo.l_scale(x + dloc, trim)
    assert l2_add == approx(l2, abs=1e-8, rel=1e-5)


@given(
    x=st_a1 | st_a2,
    trim=st_trim,
    dscale=st.floats(-1e4, -1e-2) | st.floats(1e-2, 1e4)
)
def test_l_scale_linear_scale(x, trim, dscale):
    l2 = lmo.l_scale(x, trim)
    assert np.isfinite(l2)
    assert np.isscalar(l2)
    assert round(l2, 8) >= 0

    # asymmetric trimming flips under sign change
    itrim = trim[::-1] if dscale < 0 and isinstance(trim, tuple) else trim

    l2_mul = lmo.l_scale(x * dscale, itrim)
    assert l2_mul == approx(abs(l2 * dscale), abs=1e-5, rel=1e-3)


@given(a=st.floats(0.1, 10))
def test_lm_expon(a):
    l_stats = np.array([a, a / 2, 1 / 3, 1 / 6])

    ppf = functools.partial(expon_ppf, a=a)
    cdf = functools.partial(expon_cdf, a=a)

    l_ppf = lmo.l_moment_from_ppf(ppf, [0, 1, 2, 3, 4])
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats)

    l_cdf = lmo.l_moment_from_cdf(cdf, [0, 1, 2, 3, 4], support=(0, np.inf))
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats)


def test_lm_normal():
    # cdf and ppf of normal dist
    from statistics import NormalDist

    mu, sigma = 100, 15
    IQ = NormalDist(mu, sigma)

    l_stats = np.array([
        mu,
        sigma / np.sqrt(np.pi),
        0,
        30 * np.arctan(np.sqrt(2)) / np.pi - 9,
    ])

    l_ppf = lmo.l_moment_from_ppf(IQ.inv_cdf, [0, 1, 2, 3, 4])
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats)

    l_cdf = lmo.l_moment_from_cdf(IQ.cdf, [0, 1, 2, 3, 4])
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats)


def test_tlm_normal():
    # cdf and ppf of normal dist
    from statistics import NormalDist

    mu, sigma = 100, 15
    IQ = NormalDist(mu, sigma)

    l_stats = np.array([mu, 0.2970 * sigma, 0, 0.06248])

    l_ppf = lmo.l_moment_from_ppf(IQ.inv_cdf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats, rtol=1e-4)

    l_cdf = lmo.l_moment_from_cdf(IQ.cdf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats, rtol=1e-4)


def test_tlm_cauchy():
    l_stats = np.array([0, 0.698, 0, 0.343])

    l_ppf = lmo.l_moment_from_ppf(cauchy_ppf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_ppf = l_ppf[1:] / l_ppf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_ppf, l_stats, rtol=1e-3)

    l_cdf = lmo.l_moment_from_cdf(cauchy_cdf, [0, 1, 2, 3, 4], trim=(1, 1))
    l_stats_cdf = l_cdf[1:] / l_cdf[[0, 0, 2, 2]]

    assert np.allclose(l_stats_cdf, l_stats, rtol=1e-3)


