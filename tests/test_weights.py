from hypothesis import given, strategies as st
import numpy as np

# noinspection PyProtectedMember
from lmo._utils import expand_trimming
from lmo.weights import l_weights, tl_weights, reweight


st_n = st.integers(32, 1024)
st_r = st.integers(1, 8)

st_t = st.integers(0, 4)
_st_t1 = st.integers(1, 4)
st_trim = st_t | st.tuples() | st.tuples(st_t) | st.tuples(st_t, st_t)
st_trim1 = _st_t1 | st.tuples(_st_t1) | st.tuples(_st_t1, _st_t1)


@given(n=st_n, r=st_r)
def test_a_l_ias(n, r):
    # gotta make testing fun somehow...
    w_l = l_weights(n, r)
    w_tl = tl_weights(n, r, 0)

    assert np.array_equal(w_l, w_tl)


@given(n=st_n, r=st_r, trim=st_trim)
def test_tl_basic(n, r, trim):
    w = tl_weights(n, r, trim)

    assert w.shape == (n,)
    assert np.all(np.isfinite(n))


@given(n=st_n, r=st_r, t=st_t)
def test_tl_symmetry(n, r, t):
    w = tl_weights(n, r, (t, t))

    assert np.allclose(w, w[::-1] * (-1)**(r-1))


@given(n=st_n, r=st_r, trim=st_trim1)
def test_tl_trim(n, r, trim):
    w = tl_weights(n, r, trim=trim)

    tl, tr = expand_trimming(trim)
    assert tl > 0
    assert tr > 0

    assert np.allclose(w[:tl], 0)
    assert np.allclose(w[-tr:], 0)


@given(n=st_n, trim=st_trim)
def test_tl_sum1(n, trim):
    w = tl_weights(n, 1, trim)

    assert np.allclose(np.sum(w), 1)


@given(n=st_n, r=st.integers(2, 16), trim=st_trim)
def test_tl_sum2p(n, r, trim):
    w = tl_weights(n, r, trim)

    assert np.allclose(np.sum(w), 0)


@given(n=st_n, r=st_r, trim=st_trim, const=st.floats(0.1, 10))
def test_reweight_identity(n, r, trim, const):
    w_r = tl_weights(n, r, trim)
    w_x = np.full_like(w_r, const)

    v_r = reweight(w_r, w_x)
    assert np.allclose(v_r, w_r)
