# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy.testing import assert_allclose, assert_equal

from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import SearchStrategy
from pytest import approx

import lmo


_R_MAX = 8
_T_MAX = 2
_N_MIN = _R_MAX + 2 * _T_MAX

st_r = st.integers(1, _R_MAX)
st_k = st.integers(2, _R_MAX)
st_t = st.integers(0, _T_MAX)
st_n = st.integers(_N_MIN, 50)
st_n2 = st.tuples(st_n, st.integers(1, 10))
st_trim = st.tuples(st_t, st_t)
st_dtype: SearchStrategy[np.dtype[Any]] = hnp.floating_dtypes(sizes=(64,))

__st_a_kwargs: dict[str, SearchStrategy[Any]] = {
    'dtype': st_dtype,
    'elements': st.floats(-1e4, -1e-2) | st.floats(1e-2, 1e4),
}
st_a1 = cast(
    SearchStrategy[npt.NDArray[Any]],
    hnp.arrays(shape=st_n, unique=False, **__st_a_kwargs)
)
st_a1_unique = cast(
    SearchStrategy[npt.NDArray[Any]],
    hnp.arrays(shape=st_n, unique=True, **__st_a_kwargs)
)
st_a2 = cast(
    SearchStrategy[npt.NDArray[Any]],
    hnp.arrays(shape=st_n2, unique=False, **__st_a_kwargs),
)


@given(a=st_a1, trim=st_trim)
def test_l_moment_zero(a: npt.NDArray[Any], trim: tuple[int, int]):
    l0 = lmo.l_moment(a, 0, trim)

    assert np.isscalar(l0)
    assert l0 == 1


@given(a=st_a1, r=st_r, trim=st_trim, w_const=st.floats(0.1, 10))
def test_l_moment_aweights_const(
    a: npt.NDArray[Any],
    r: int,
    trim: tuple[int, int],
    w_const: float,
):
    l_r = lmo.l_moment(a, r, trim)

    w = np.full_like(a, w_const)
    l_r_w = lmo.l_moment(a, r, trim, aweights=w)

    assert l_r_w == approx(l_r, rel=1e-5, abs=1e-8)


@given(a=st_a1, r=st_r, trim=st_trim)
def test_l_ratio_unit(
    a: npt.NDArray[Any],
    r: int,
    trim: tuple[int, int],
):
    tau = lmo.l_ratio(a, r, r, trim)

    assert tau == approx(1)


@given(a=st_a1 | st_a2)
def test_l_loc_mean(a: npt.NDArray[Any]):
    loc = a.mean(dtype=np.float64)
    l_loc = lmo.l_loc(a)

    assert l_loc.shape == loc.shape
    assert l_loc == approx(loc, rel=1e-5, abs=1e-8)


@given(a=st_a2)
def test_l_loc_mean_2d(a: npt.NDArray[Any]):
    locs = a.mean(axis=0, dtype=np.float64)
    l_locs = lmo.l_loc(a, axis=0)

    assert len(l_locs) == a.shape[1]
    assert l_locs.shape == locs.shape
    assert_allclose(l_locs, locs)

    l_locs_t = lmo.l_loc(a.T, axis=1)

    assert l_locs_t.shape == l_locs.shape
    assert_allclose(l_locs_t, l_locs)


@given(x0=st.floats(-1e6, 1e6), n=st_n, dtype=st_dtype, trim=st_trim)
def test_l_loc_const(
    x0: float,
    n: int,
    dtype: np.dtype[Any],
    trim: tuple[int, int],
):
    x = np.full(n, x0, dtype=dtype)
    l_1 = lmo.l_loc(x, trim)

    assert l_1 == approx(x0, rel=1e-5, abs=1e-8)


@given(
    x=st_a1 | st_a2,
    trim=st_trim,
    dloc=st.floats(-1e3, 1e3),
    dscale=st.floats(1e-3, 1e3),
)
def test_l_loc_linearity(
    x: npt.NDArray[Any],
    trim: tuple[int, int],
    dloc: float,
    dscale: float,
):
    l1 = lmo.l_loc(x, trim)
    assert np.isfinite(l1)
    assert np.isscalar(l1)

    l1_add = lmo.l_loc(x + dloc, trim)
    assert l1_add == approx(l1 + dloc, rel=1e-5, abs=1e-8)

    l1_mul = lmo.l_loc(x * dscale, trim)
    assert l1_mul == approx(l1 * dscale, rel=1e-5, abs=1e-8)


@given(a=st_a1)
def test_l_scale_equiv_md(a: npt.NDArray[Any]):
    # half gini/abs mean difference (MD)
    n = len(a)
    scale = abs(a - a[:, None]).mean() / (2 - 2 / n)

    l2 = lmo.l_scale(a)

    assert l2.shape == scale.shape
    assert l2 == approx(scale, rel=1e-5, abs=1e-8)


@given(x0=st.floats(-1e6, 1e6), n=st_n, dtype=st_dtype, trim=st_trim)
def test_l_scale_const(
    x0: float,
    n: int,
    dtype: np.dtype[Any],
    trim: tuple[int, int],
):
    x = np.full(n, x0, dtype=dtype)
    l2 = lmo.l_scale(x, trim)
    assert l2 == approx(0, rel=1e-5, abs=1e-8)


@given(x=st_a1 | st_a2, trim=st_trim, dloc=st.floats(-1e3, 1e3))
def test_l_scale_invariant_loc(
    x: npt.NDArray[Any],
    trim: tuple[float, float],
    dloc: float
):
    l2 = lmo.l_scale(x, trim)
    assert np.isfinite(l2)
    assert np.isscalar(l2)
    assert round(l2, 8) >= 0

    l2_add = lmo.l_scale(x + dloc, trim)
    assert l2_add == approx(l2, rel=1e-5, abs=1e-8)


@given(
    x=st_a1 | st_a2,
    trim=st_trim,
    dscale=st.floats(-1e2, -1e-2) | st.floats(1e-2, 1e2),
)
def test_l_scale_linear_scale(
    x: npt.NDArray[Any],
    trim: tuple[int, int],
    dscale: float,
):
    l2 = lmo.l_scale(x, trim)
    assert np.isfinite(l2)
    assert np.isscalar(l2)
    assert round(l2, 8) >= 0

    # asymmetric trimming flips under sign change
    itrim = trim[::-1] if dscale < 0 else trim

    l2_mul = lmo.l_scale(x * dscale, itrim)
    assert l2_mul == approx(np.abs(l2 * dscale), abs=1e-8)
