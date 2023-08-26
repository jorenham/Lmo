# type: ignore

import numpy as np
from hypothesis import (
    given,
    strategies as st,
)
from lmo._lm import l_weights

MAX_N = 1 << 10
MAX_R = 8
MAX_T = 4

st_n = st.integers(MAX_R + MAX_T * 2 + 1, MAX_N)
st_r = st.integers(1, MAX_R)

st_t_f = st.floats(0, MAX_T, exclude_min=True)
st_t_i = st.integers(1, MAX_T)
st_t_i0 = st.integers(0, MAX_T)

st_trim_i = st.tuples(st_t_i, st_t_i)
st_trim_i0 = st.tuples(st_t_i0, st_t_i0)


@given(n=st_n, r=st_r, trim0=st.just((0, 0)))
def test_l_weights_alias(n, r, trim0):
    w_l = l_weights(r, n)
    w_tl = l_weights(r, n, trim0)

    assert np.array_equal(w_l, w_tl)


@given(n=st_n, r=st_r, trim=st_trim_i0)
def test_l_weights_basic(n, r, trim):
    w = l_weights(r, n, trim)

    assert w.shape == (r, n)
    assert np.all(np.isfinite(n))
    assert w.dtype.type is np.float_


# symmetries only apply for symmetric trimming, for obvious reasons
@given(n=st_n, t=st_t_i0)
def test_l_weights_symmetry(n, t):
    w = l_weights(MAX_R, n, (t, t))

    w_evn_lhs, w_evn_rhs = w[::2], w[::2, ::-1]
    assert np.allclose(w_evn_lhs, w_evn_rhs)

    w_odd_lhs, w_odd_rhs = w[1::2], w[1::2, ::-1]
    assert np.allclose(w_odd_lhs, -w_odd_rhs)


def test_l_weights_symmetry_large_even_r():
    w = l_weights(16, MAX_N * 2)

    w_evn_lhs, w_evn_rhs = w[::2], w[::2, ::-1]
    assert np.allclose(w_evn_lhs, w_evn_rhs)


@given(n=st_n, r=st_r, trim=st_trim_i)
def test_l_weights_trim(n, r, trim):
    w = l_weights(r, n, trim)

    tl, tr = trim
    assert tl > 0
    assert tr > 0

    assert np.allclose(w[:, :tl], 0)
    assert np.allclose(w[:, n - tr :], 0)


@given(n=st_n, r=st.integers(2, MAX_R), trim=st_trim_i0)
def test_tl_weights_sum(n, r, trim):
    w = l_weights(r, n, trim)
    w_sum = w.sum(axis=-1)

    assert np.allclose(w_sum, np.eye(r, 1).ravel())
