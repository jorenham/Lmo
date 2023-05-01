from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np

import lmo

_max_k = 8

st_elems = st.floats(allow_nan=False, allow_infinity=False, width=16)
st_k = st.integers(min_value=1, max_value=_max_k)
st_n = st.integers(min_value=_max_k + 1, max_value=100)
st_m = st.integers(min_value=1, max_value=10)
st_arrays_1d = hnp.arrays(
    dtype=hnp.floating_dtypes(),
    shape=st_n,
    elements=st_elems
)
st_arrays_2d = hnp.arrays(
    dtype=hnp.floating_dtypes(),
    shape=st.tuples(st_n, st_m),
    elements=st_elems
)


@given(a=st_arrays_1d, k_max=st_k)
def test_moments_1d(a: np.ndarray, k_max: int):
    l_k = lmo.l_moments(a, k_max)

    assert l_k.shape == (1 + k_max,)
    assert l_k[0] == 1


@given(a=st_arrays_2d, k_max=st_k)
def test_moments_2d(a: np.ndarray, k_max: int):
    l_k0 = lmo.l_moments(a[:, 0], k_max)
    l_km = lmo.l_moments(a, k_max)

    assert l_km.shape == (1 + k_max, a.shape[1])
    assert l_km.dtype.type is a.dtype.type
    assert np.all(l_km[0] == 1)
    assert np.allclose(l_km[:, 0], l_k0)


@given(a=st_arrays_1d)
def test_moment0_1d(a: np.ndarray):
    l0 = lmo.l_moment(a, 0)

    assert np.isscalar(l0)
    assert l0 == 1


@given(a=st_arrays_2d)
def test_moment0_2d(a: np.ndarray):
    l0 = lmo.l_moment(a, 0)

    assert not np.isscalar(l0)
    assert l0.shape == (len(l0),)
    assert l0.dtype.type == a.dtype.type
    assert np.all(l0 == 1)


@given(a=st_arrays_1d | st_arrays_2d)
def test_loc(a: np.ndarray):
    loc = a.mean(axis=0)
    l_loc = lmo.l_loc(a)

    assert l_loc.shape == loc.shape
    assert l_loc.dtype.type is a.dtype.type

    assert np.allclose(l_loc, loc, rtol=1e-3)


@given(a=st_arrays_1d | st_arrays_2d)
def test_scale(a: np.ndarray):
    # half mean absolute difference
    scale = np.mean(
        abs(a - a[:, None]),
        axis=tuple(range(a.ndim)) if a.ndim > 1 else None,
        dtype=a.dtype
    ) / 2

    l_scale = lmo.l_scale(a)

    assert l_scale.shape == scale.shape
    assert l_scale.dtype.type is a.dtype.type

    assert np.allclose(l_scale, scale)
