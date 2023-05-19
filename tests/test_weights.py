from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp
from pytest import approx

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

@given(n=st_n, r=st_r, trim0=st.just(0) | st.just((0, 0)))
def test_l_weights_alias(n, r, trim0):
    w_l = l_weights(n, r)
    w_tl = tl_weights(n, r, trim0)

    assert np.array_equal(w_l, w_tl)


@given(n=st_n, r=st_r, dtype=st_dtype_float)
def test_l_weights_dtype(n, r, dtype):
    w_r = tl_weights(n, r, dtype=dtype)

    assert w_r.dtype.type is np.dtype(dtype).type


@given(n=st_n, r=st_r, trim=st_trim, dtype=st_dtype_float)
def test_tl_weights_dtype(n, r, trim, dtype):
    w_r = tl_weights(n, r, trim, dtype=dtype)

    assert w_r.dtype.type is np.dtype(dtype).type


@given(n=st_n, r=st_r, trim=st_trim)
def test_tl_weights_basic(n, r, trim):
    w = tl_weights(n, r, trim)

    assert w.shape == (n,)
    assert np.all(np.isfinite(n))
    assert w.dtype.type is np.float_


@given(n=st_n, r=st_r, t=st_t)
def test_tl_weights_symmetry(n, r, t):
    w = tl_weights(n, r, (t, t))

    assert np.allclose(w, w[::-1] * (-1)**(r-1))


@given(n=st_n, r=st_r, trim=st_trim1)
def test_tl_weights_trim(n, r, trim):
    w = tl_weights(n, r, trim=trim)

    tl, tr = expand_trimming(trim)
    assert tl > 0
    assert tr > 0

    assert np.allclose(w[:tl], 0)
    assert np.allclose(w[-tr:], 0)


@given(n=st_n, trim=st_trim)
def test_tl_weights_sum1(n, trim):
    w = tl_weights(n, 1, trim)

    assert np.allclose(np.sum(w), 1)


@given(n=st_n, r=st.integers(2, 16), trim=st_trim)
def test_tl_weights_sum2p(n, r, trim):
    w = tl_weights(n, r, trim)

    assert np.allclose(np.sum(w), 0)

@given(n=st_n, r=st_r, trim=st_trim, dtype=st_dtype_float, dtype2=st_dtype_real)
def test_reweight_dtype(n, r, trim, dtype, dtype2):
    w_r = tl_weights(n, r, trim, dtype=dtype)
    v_r = reweight(w_r, np.ones(n, dtype=dtype2))

    assert w_r.dtype.type is np.dtype(dtype).type
    assert v_r.dtype == w_r.dtype
    assert v_r.dtype.type is w_r.dtype.type


@given(n=st_n, r=st_r, trim=st_trim, const=st.floats(0.01, 100))
def test_reweight_identity(n, r, trim, const):
    w_r = tl_weights(n, r, trim)
    w_x = np.full_like(w_r, const)

    v_r = reweight(w_r, w_x)

    assert np.allclose(v_r, w_r)


@given(n=st_n, r=st_r, trim=st_trim, ri=st.floats(0, 1))
def test_reweight_censor(n, r, trim, ri):
    w_r = tl_weights(n, r, trim=trim)

    # pick index to censor within the non-trimmed TL-weights
    tl, tr = expand_trimming(trim)
    i = tl + round((n - tl - tr - 1) * ri)

    w_x = np.ones(n)
    w_x[i] = 0.0

    v_r = reweight(w_r, w_x)

    assert v_r[i] == 0
    assert v_r.sum() == approx(w_r.sum())
