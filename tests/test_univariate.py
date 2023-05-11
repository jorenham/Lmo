from datetime import timedelta

from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np

import lmo
from lmo.stats import tl_ratio_max

_R_MAX = 8
_S_MAX = _T_MAX = 2
_N_MIN = _R_MAX + _S_MAX + _T_MAX

st_r = st.integers(1, _R_MAX)
st_k = st.integers(2, _R_MAX)
st_s = st.integers(0, _S_MAX)
st_t = st.integers(0, _T_MAX)

__st_a_kwargs = {
    'dtype': hnp.floating_dtypes(
        sizes=(32, 64, 128) if hasattr(np, 'float128') else (32, 64)
    ),
    'elements': st.floats(-(1 << 20), 1 << 20, width=32),
}
st_shape_a1 = st.integers(_N_MIN, 50)
st_a1 = hnp.arrays(shape=st_shape_a1, **__st_a_kwargs)
st_a1_unique = hnp.arrays(shape=st_shape_a1, unique=True, **__st_a_kwargs)

st_a2 = hnp.arrays(
    shape=st.tuples(st_shape_a1, st.integers(1, 10)),
    **__st_a_kwargs
)


@given(a=st_a1)
def test_tl_moment_zero(a: np.ndarray):
    l0 = lmo.tl_moment(a, 0)

    assert np.isscalar(l0)
    assert l0 == 1


@given(a=st_a1, r=st_r, s=st_s, t=st_t)
def test_tl_ratio_unit(a: np.ndarray, r: int, s: int, t: int):
    tau = lmo.tl_ratio(a, r, r, s, t)

    assert np.allclose(tau, 1)


@given(a=st_a1, s=st_s, t=st_t)
def test_tl_cv_bound(a: np.ndarray,  s: int, t: int):
    """Theorem 2 in J.R.M. Hosking (1990), but exended for TL moments."""
    tl_cv_max = tl_ratio_max(2, 1, s, t)

    a = np.abs(a) + 0.1  # ensure positive and nonzero mean
    tl_cv = lmo.tl_ratio(a, 2, 1, s, t)

    # nan is "fine" too
    assert tl_cv <= tl_cv_max


# noinspection PyArgumentEqualDefault
@settings(deadline=timedelta(seconds=1))
@given(a=st_a1_unique, r=st.integers(3, _R_MAX), s=st_s, t=st_t)
def test_tl_ratio_bound(a: np.ndarray, r: int, s: int, t: int):
    tau_max = tl_ratio_max(r, 2, s, t)
    tau = lmo.tl_ratio(a, r, 2, s, t)

    # nan is "fine" too
    assert abs(tau) <= tau_max + 1e-8


@given(a=st_a1 | st_a2)
def test_l_loc(a: np.ndarray):
    loc = a.mean(dtype=np.float_)
    l_loc = lmo.l_loc(a)

    assert l_loc.shape == loc.shape
    assert np.allclose(l_loc, loc, rtol=1e-4)


@given(a=st_a2)
def test_l_loc_2d(a: np.ndarray):
    locs = a.mean(axis=0, dtype=np.float_)
    l_locs = lmo.l_loc(a, axis=0)

    assert len(l_locs) == a.shape[1]
    assert l_locs.shape == locs.shape
    assert np.allclose(l_locs, locs, rtol=1e-4)

    l_locs_t = lmo.l_loc(a.T, axis=1)

    assert l_locs_t.shape == l_locs.shape
    assert np.allclose(l_locs_t, l_locs)


@given(a=st_a1)
def test_l_scale(a: np.ndarray):
    # half mean absolute difference
    n = len(a)
    scale = abs(a - a[:, None]).mean() / (2 - 2 / n)

    l_scale = lmo.l_scale(a)

    assert l_scale.shape == scale.shape
    assert np.allclose(l_scale, scale, rtol=1e-4)


# TODO: tl_loc, tl_scale, (t)l_skew, (t)l_kurt
