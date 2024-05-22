import numpy as np
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra import numpy as hnp

from lmo._utils import ordered


st_n = st.integers(2, 50)
st_x1 = hnp.arrays(shape=st_n, dtype=np.float64, elements=st.floats(-10, 10))


@given(x=st_x1)
def test_order_stats_sorted(x):
    x_k = ordered(x)

    assert x_k.shape == x.shape
    assert np.all(x_k[:-1] <= x_k[1:])


@given(x=st_x1)
def test_order_stats_sorted_xx(x):
    x_k = ordered(x, x)

    assert x_k.shape == x.shape
    assert np.all(x_k[:-1] <= x_k[1:])


@given(x=st_x1)
def test_order_stats_sorted_concomitant(x):
    x_k = ordered(x, -x)

    assert x_k.shape == x.shape
    assert np.all(x_k[:-1] >= x_k[1:])


@given(x=st_x1)
def test_order_stats_sorted_concomitant_2d(x):
    x_mn = np.stack((x, x + 1))
    x_mk = ordered(x_mn, -x, axis=-1)

    assert x_mk.shape == x_mn.shape
    assert np.all(x_mk[:, :-1] >= x_mk[:, 1:])

    x_nm = x_mn.T
    x_km = ordered(x_mn.T, -x, axis=0)

    assert x_km.shape == x_nm.shape
    assert np.all(x_km[:-1] >= x_km[1:])
    assert np.allclose(x_km, x_mk.T)


@given(x=st_x1, f=st.integers(1, 100))
def test_order_stats_fweights_const(x, f):
    fweights = np.full(x.shape, f, dtype=np.int_)

    x_k = ordered(x)
    x_l = ordered(x, fweights=fweights)

    assert x_k.shape == x_l.shape
    assert np.all(x_k == x_l)


@given(x=st_x1, a=st.floats(0.01, 100))
def test_order_stats_aweights_const(x, a):
    aweights = np.full(x.shape, a)

    x_k = ordered(x)
    x_l = ordered(x, aweights=aweights)

    assert x_k.shape == x_l.shape
    assert np.allclose(x_k, x_l)


@given(x=st_x1)
def test_order_stats_fweights_remove(x):
    fweights = np.ones(x.shape, dtype=np.int_)
    fweights[0] = 0

    x_k = ordered(x[1:])
    x_l = ordered(x, fweights=fweights)

    assert len(x_l) == len(x_k)
    assert np.all(x_l == x_k)


@given(x=st_x1)
def test_order_stats_fweights_double(x):
    fweights = np.ones(x.shape, dtype=np.int_)
    fweights[0] = 2

    x_k = ordered(np.r_[x[0], x])
    x_l = ordered(x, fweights=fweights)

    assert len(x_l) == len(x_k)
    assert np.all(x_l == x_k)
