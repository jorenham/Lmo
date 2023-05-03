from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np

import lmo

_max_r = 8

st_elems = st.floats(allow_nan=False, allow_infinity=False, width=16)
st_r = st.integers(min_value=1, max_value=_max_r)
st_n = st.integers(min_value=_max_r + 1, max_value=100)
st_m = st.integers(min_value=1, max_value=10)
st_s = st.integers(0, 4)
st_t = st.integers(0, 4)
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


@given(a=st_arrays_1d)
def test_moment0_1d(a: np.ndarray):
    l0 = lmo.tl_moment(a, 0)

    assert np.isscalar(l0)
    assert l0 == 1


@given(a=st_arrays_2d)
def test_moment0_2d(a: np.ndarray):
    l0 = lmo.tl_moment(a, 0, axis=1)

    assert not np.isscalar(l0)
    assert l0.shape == (len(l0),)
    assert l0.dtype.type == a.dtype.type
    assert np.all(l0 == 1)


@given(a=st_arrays_1d | st_arrays_2d)
def test_l_loc(a: np.ndarray):
    loc = a.mean(axis=0, dtype=np.float_)
    l_loc = lmo.l_loc(a, axis=0)

    assert l_loc.shape == loc.shape
    assert np.allclose(l_loc, loc, rtol=1e-3)


@given(a=st_arrays_1d | st_arrays_2d)
def test_l_scale(a: np.ndarray):
    # half mean absolute difference
    n = len(a)
    scale = np.sum(
        abs(a - a[:, None]),
        axis=tuple(range(a.ndim)) if a.ndim > 1 else None,
        dtype=np.float_
    ) / (n**2 - n) / 2

    l_scale = lmo.l_scale(a, axis=0)

    assert l_scale.shape == scale.shape
    assert np.allclose(l_scale.astype(a.dtype), scale.astype(a.dtype))
