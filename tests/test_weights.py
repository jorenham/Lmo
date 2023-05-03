from hypothesis import given, strategies as st
import numpy as np

from lmo.weights import l_weights, tl_weights


st_n = st.integers(32, 1024)
st_r = st.integers(1, 8)
st_s = st.integers(0, 4)
st_s1 = st.integers(1, 4)
st_t = st.integers(0, 4)
st_t1 = st.integers(1, 4)


@given(n=st_n, r=st_r)
def test_a_l_ias(n, r):
    # gotta make testing fun somehow...
    w_l = l_weights(n, r)
    w_tl = tl_weights(n, r, 0, 0)

    assert np.array_equal(w_l, w_tl)


@given(n=st_n, r=st_r, s=st_s, t=st_t)
def test_tl_basic(n, r, s, t):
    w = tl_weights(n, r, s, t)

    assert w.shape == (n,)
    assert np.all(np.isfinite(n))


@given(n=st_n, r=st_r, t=st_t)
def test_tl_symmetry(n, r, t):
    w = tl_weights(n, r, t, t)

    assert np.allclose(w, w[::-1] * (-1)**(r-1))


@given(n=st_n, r=st_r, s=st_s1, t=st_t1)
def test_tl_trim(n, r, s, t):
    w = tl_weights(n, r, s, t)

    assert np.allclose(w[:s], 0)
    assert np.allclose(w[-t:], 0)


@given(n=st_n, s=st_s, t=st_t)
def test_tl_sum1(n, s, t):
    w = tl_weights(n, 1, s, t)

    assert np.allclose(np.sum(w), 1)


@given(n=st_n, r=st.integers(2, 16), s=st_s, t=st_t)
def test_tl_sum2p(n, r, s, t):
    w = tl_weights(n, r, s, t)

    assert np.allclose(np.sum(w), 0)
