from datetime import timedelta

from pytest import approx
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np

import lmo
from lmo.stats import l_ratio_max

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


@given(a=st_a1, trim=st_trim)
def test_l_variation_bound(a,  trim):
    """Theorem 2 in J.R.M. Hosking (1990), but exended for TL moments."""
    tl_cv_max = l_ratio_max(2, 1, trim)

    a = np.abs(a) + 0.1  # ensure positive and nonzero mean
    tl_cv = lmo.l_variation(a, trim)

    # nan is "fine" too
    assert tl_cv <= tl_cv_max


# noinspection PyArgumentEqualDefault
@settings(deadline=timedelta(seconds=1))
@given(a=st_a1_unique, r=st.integers(3, 6), trim=st_trim)
def test_l_ratio_bound(a, r, trim):
    tau_max = l_ratio_max(r, 2, trim=trim)
    tau = lmo.l_ratio(a, r, 2, trim=trim)

    assert abs(tau) <= tau_max + tau_max * 1e-5


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
    assert l2_mul == approx(abs(l2 * dscale), abs=1e-8, rel=1e-5)
