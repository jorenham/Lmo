import functools
from typing import Any

import numpy as np
import pytest
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra import numpy as hnp
from numpy.testing import (
    assert_allclose as _assert_allclose,
    assert_array_equal,
)

from lmo import l_weights
from .conftest import tmp_cache


# matches np.allclose
assert_allclose = functools.partial(_assert_allclose, rtol=1e-5, atol=1e-8)


MAX_R = 8
MAX_T = 4
MIN_N = MAX_R + MAX_T * 2 + 1
MAX_N = 1 << 8


st_n = st.integers(MIN_N, MAX_N)
st_r = st.integers(1, MAX_R)

st_i_eq0 = st.just(0)
st_i_ge0 = st.integers(0, MAX_T)
st_i_gt0 = st.integers(1, MAX_T)

st_i2_eq0 = st.tuples(st.just(0), st.just(0))
st_i2_ge0 = st.tuples(st.integers(0, MAX_T), st.integers(0, MAX_T))
st_i2_gt0 = st.tuples(st.integers(1, MAX_T), st.integers(1, MAX_T))

st_i12_eq0 = st_i_eq0 | st_i2_eq0
st_i12_ge0 = st_i_ge0 | st_i2_ge0
st_i12_gt0 = st_i_gt0 | st_i2_gt0

st_floating = hnp.floating_dtypes()


@given(n=st_n, trim=st_i12_eq0)
def test_empty(n: int, trim: int | tuple[int, int]):
    w = l_weights(0, n, trim)
    assert w.shape == (0, n)


@given(n=st_n, r=st_r, trim=st_i12_eq0)
def test_untrimmed(n: int, r: int, trim: int | tuple[int, int]):
    w_l = l_weights(r, n)
    w_tl = l_weights(r, n, trim)

    assert_array_equal(w_l, w_tl)


@given(n=st_n, r=st_r, trim=st_i12_ge0)
def test_default(n: int, r: int, trim: int | tuple[int, int]):
    w = l_weights(r, n, trim)

    assert w.shape == (r, n)
    assert np.all(np.isfinite(w))
    assert w.dtype.type is np.float64


@given(n=st_n, r=st_r, trim=st_i12_ge0, dtype=st_floating)
def test_dtype(
    n: int,
    r: int,
    trim: int | tuple[int, int],
    dtype: np.dtype[np.floating[Any]],
):
    w = l_weights(r, n, trim, dtype=dtype)

    assert np.all(np.isfinite(w))
    assert w.dtype.type is dtype.type


@given(n=st_n, t=st_i_ge0)
def test_symmetry(n: int, t: int):
    w = l_weights(MAX_R, n, (t, t))

    w_evn_lhs, w_evn_rhs = w[::2], w[::2, ::-1]
    assert_allclose(w_evn_lhs, w_evn_rhs)

    w_odd_lhs, w_odd_rhs = w[1::2], w[1::2, ::-1]
    assert_allclose(w_odd_lhs, -w_odd_rhs)


def test_l_weights_symmetry_large_even_r():
    w = l_weights(16, MAX_N * 2)

    w_evn_lhs, w_evn_rhs = w[::2], w[::2, ::-1]
    assert_allclose(w_evn_lhs, w_evn_rhs)


@given(n=st_n, r=st_r, trim=st_i2_gt0)
def test_trim(n: int, r: int, trim: tuple[int, int]):
    w = l_weights(r, n, trim)

    tl, tr = trim
    assert tl > 0
    assert tr > 0

    assert_allclose(w[:, :tl], 0)
    assert_allclose(w[:, n - tr :], 0)


@given(n=st_n, r=st.integers(2, MAX_R), trim=st_i12_ge0)
def test_sum(n: int, r: int, trim: int | tuple[int, int]):
    w = l_weights(r, n, trim)
    w_sum = w.sum(axis=-1)

    assert_allclose(w_sum, np.eye(r, 1).ravel())


@given(n=st_n, r=st.integers(4, MAX_R), trim=st_i12_ge0)
def test_uncached(n: int, r: int, trim: int | tuple[int, int]):
    with tmp_cache() as cache:
        w0 = l_weights(r, n, trim, cache=False)
        w1 = l_weights(r, n, trim, cache=False)

        assert not cache
        assert w0 is not w1
        assert_array_equal(w0, w1)


@given(n=st_n, r=st.integers(4, MAX_R), trim=st_i12_ge0)
def test_cached(n: int, r: int, trim: int | tuple[int, int]):
    cache_key = (n, *trim) if isinstance(trim, tuple) else (n, trim, trim)

    with tmp_cache() as cache:
        assert cache_key not in cache

        w0 = l_weights(r, n, trim, cache=True, dtype=np.longdouble)
        assert cache_key in cache
        w0_cached = cache[cache_key]

        # cached weights should be readonly
        w0_orig = w0[0, 0]
        with pytest.raises(
            ValueError,
            match='assignment destination is read-only',
        ):
            w0[0, 0] = w0_orig + 1
        assert w0[0, 0] == w0_orig

        w1 = l_weights(r, n, trim, cache=True, dtype=np.longdouble)
        w1_cached = cache[cache_key]
        assert w0_cached is w1_cached

        # this requires `r>=4`, `dtype=np.longdouble` and `r == r_cached`
        assert w0 is w1
