# type: ignore

from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

import numpy as np

from lmo._lm import l_weights

st_n = st.integers(32, 1024)
st_r = st.integers(1, 8)

st_t = st.integers(0, 4)
_st_t1 = st.integers(1, 4)
st_trim = st.tuples(st_t, st_t)
st_trim1 = st.tuples(_st_t1, _st_t1)

st_dtype_float = hnp.floating_dtypes(
    # system dependent, e.g. numpy.float128 is not always available
    sizes=[t().nbytes << 3 for t in np.sctypes['float']]
)
st_dtype_real = st.one_of(
    hnp.boolean_dtypes(),
    hnp.integer_dtypes(),
    hnp.unsigned_integer_dtypes(),
    hnp.floating_dtypes(),
)

@given(n=st_n, r=st_r, trim0=st.just((0, 0)))
def test_l_weights_alias(n, r, trim0):
    w_l = l_weights(r, n)
    w_tl = l_weights(r, n, trim0)

    assert np.array_equal(w_l, w_tl)


@given(n=st_n, r=st_r, trim=st_trim, dtype=st_dtype_float)
def test_l_weights_dtype(n, r, trim, dtype):
    w_r = l_weights(r, n, trim, dtype=dtype)

    assert w_r.dtype.type is np.dtype(dtype).type


@given(n=st_n, r=st_r, trim=st_trim)
def test_l_weights_basic(n, r, trim):
    w = l_weights(r, n, trim)

    assert w.shape == (r, n)
    assert np.all(np.isfinite(n))
    assert w.dtype.type is np.float_


@given(n=st_n, r=st_r, t=st_t)
def test_l_weights_symmetry(n, r, t):
    w = l_weights(r, n, (t, t))

    assert np.allclose(w[::2], w[::2, ::-1])
    assert np.allclose(w[1::2], -w[1::2, ::-1])


@given(n=st_n, r=st_r, trim=st_trim1)
def test_l_weights_trim(n, r, trim):
    w = l_weights(r, n, trim)

    tl, tr = trim
    assert tl > 0
    assert tr > 0

    assert np.allclose(w[:, :tl], 0)
    assert np.allclose(w[:, n-tr:], 0)


@given(n=st_n, r=st.integers(2, 16), trim=st_trim)
def test_tl_weights_sum(n, r, trim):
    w = l_weights(r, n, trim)

    assert np.allclose(np.sum(w[0]), 1)
    assert np.allclose(np.sum(w[1:]), 0)
