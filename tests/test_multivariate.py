from datetime import timedelta

import numpy as np
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from hypothesis.extra import numpy as hnp

import lmo

_SEED = 12345

_R_MAX = 8
_S_MAX = _T_MAX = 2
_N_MIN = _R_MAX + _S_MAX + _T_MAX

st_r = st.integers(1, _R_MAX)
st_k = st.integers(2, _R_MAX)
st_t = st.integers(0, _T_MAX)
st_trim = st.tuples(st_t, st_t)

__st_a_kwargs = {
    "dtype": hnp.floating_dtypes(
        sizes=(64, 128) if hasattr(np, "float128") else (64,),
    ),
    "elements": st.floats(-(1 << 20), 1 << 20),
}

st_m = st.integers(1, 5)
st_n = st.integers(_N_MIN, 50)
st_mn = st.tuples(st_m, st_n)
st_a = hnp.arrays(shape=st_mn, **__st_a_kwargs)
st_a_unique = hnp.arrays(shape=st_mn, unique=True, **__st_a_kwargs)


@given(r=st_r, n=st_n, trim=st_trim)
def test_l_comoment_empty(r: int, n: int, trim):
    l_00 = lmo.l_comoment(np.empty((0, n)), r, trim)

    assert l_00.shape == (0, 0)


@given(a=st_a, trim=st_trim)
def test_l_comoment_zero(a: np.ndarray, trim):
    l_aa = lmo.l_comoment(a, 0, trim)

    assert l_aa.shape == (len(a), len(a))
    assert np.array_equal(l_aa, np.eye(len(a)))


@given(a=st_a, r=st_r, trim=st_trim)
def test_tl_comoment_rowvar(a: np.ndarray, r: int, trim):
    l_aa = lmo.l_comoment(a, r, trim)
    l_aa_t = lmo.l_comoment(a.T, r, trim, rowvar=False)

    assert np.array_equal(l_aa, l_aa_t)


@given(a=st_a, r=st_r, trim=st_trim)
def test_tl_comoment_diag(a: np.ndarray, r: int, trim):
    l_a = lmo.l_moment(a, r, trim, axis=1)
    L_aa = lmo.l_comoment(a, r, trim)

    assert np.allclose(L_aa.diagonal(), l_a)


@given(a=st_a, r=st_r, trim=st_trim)
def test_l_comoment_rowwise(a: np.ndarray, r: int, trim):
    l_a = lmo.l_moment(a, r, trim, axis=1)

    def func(a_m):
        return lmo.l_comoment(a_m[None, :], r, trim)[0, 0]

    L_a1 = np.apply_along_axis(func, 1, a)

    assert L_a1.shape == l_a.shape
    assert np.allclose(L_a1, l_a)


@given(a=st_a)
def test_l_coloc_mean(a: np.ndarray):
    m_a = a.mean(1)
    l_aa = lmo.l_coloc(a)
    l_a0 = l_aa[:, 0]

    assert np.allclose(l_a0, m_a, atol=1e-3, rtol=1e-3)


@settings(deadline=timedelta(seconds=1))
@given(a=st_a_unique)
def test_l_corr_standard(a: np.ndarray):
    r_aa = lmo.l_corr(a)

    assert np.all(r_aa.diagonal() == 1)
    assert np.all(r_aa <= 1)
