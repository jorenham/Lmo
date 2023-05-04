from datetime import timedelta

from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np

import lmo

_R_MAX = 8
_S_MAX = _T_MAX = 2
_N_MIN = _R_MAX + _S_MAX + _T_MAX

st_r = st.integers(1, _R_MAX)
st_k = st.integers(2, _R_MAX)
st_s = st.integers(0, _S_MAX)
st_t = st.integers(0, _T_MAX)

__st_a_kwargs = {
    'dtype': hnp.floating_dtypes(
        sizes=(32, 64, 128) if hasattr(np, 'float128') else (32, 64)
    ),
    'elements': st.floats(-(1 << 20), 1 << 20, width=32),
}

st_m = st.integers(1, 5)
st_n = st.integers(_N_MIN, 50)
st_mn = st.tuples(st_m, st_n)
st_a = hnp.arrays(shape=st_mn, **__st_a_kwargs)
st_a_unique = hnp.arrays(shape=st_mn, unique=True, **__st_a_kwargs)


@given(r=st_r, s=st_s, t=st_t)
def test_tl_comoment_empty(r: int, s: int, t: int):
    l_00 = lmo.tl_comoment(np.empty((0, 0)), r, s, t)

    assert l_00.shape == (0, 0)


@given(a=st_a, s=st_s, t=st_t)
def test_tl_comoment_zero(a: np.ndarray, s: int, t: int):
    l_aa = lmo.tl_comoment(a, 0, s, t)

    assert l_aa.shape == (len(a), len(a))
    assert np.array_equal(l_aa, np.eye(len(a)))


@given(a=st_a, r=st_r, s=st_s, t=st_t)
def test_tl_comoment_rowvar(a: np.ndarray, r: int, s: int, t: int):
    l_aa = lmo.tl_comoment(a, r, s, t)
    l_aa_t = lmo.tl_comoment(a.T, r, s, t, rowvar=False)

    assert np.all(l_aa == l_aa_t)


@given(a=st_a, r=st_r, s=st_s, t=st_t)
def test_tl_comoment_diag(a: np.ndarray, r: int, s: int, t: int):
    l_a = lmo.tl_moment(a, r, s, t, axis=1)
    L_aa = lmo.tl_comoment(a, r, s, t)

    assert np.allclose(L_aa.diagonal(), l_a)


@given(a=st_a, r=st_r, s=st_s, t=st_t)
def test_tl_comoment_rowwise(a: np.ndarray, r: int, s: int, t: int):
    l_a = lmo.tl_moment(a, r, s, t, axis=1)

    def func(a_m):
        return lmo.tl_comoment(a_m[None, :], r, s, t)[0, 0]

    L_a1 = np.apply_along_axis(func, 1, a)

    assert L_a1.shape == l_a.shape
    assert np.allclose(L_a1, l_a)


@given(a=st_a)
def test_l_coloc_mean(a: np.ndarray):
    m_i = lmo.l_coloc(a)
    assert np.allclose(m_i[:, 0], a.mean(1))


@settings(deadline=timedelta(seconds=1))
@given(a=st_a_unique)
def test_l_corr_standard(a: np.ndarray):
    r_aa = lmo.l_corr(a)

    assert np.all(r_aa.diagonal() == 1)
    assert np.all(r_aa <= 1)
