from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt

from .typing import np as lnpt
from .typing.compat import TypeVar


if TYPE_CHECKING:
    from .typing import AnyAWeights, AnyFWeights, AnyOrder, AnyOrderND, AnyTrim

__all__ = (
    'clean_order',
    'clean_orders',
    'clean_trim',
    'ensure_axis_at',
    'l_stats_orders',
    'moments_to_ratio',
    'moments_to_stats_cov',
    'sort_maybe',
    'ordered',
    'plotting_positions',
    'round0',
)


_T_scalar = TypeVar('_T_scalar', bound=np.generic)
_T_number = TypeVar('_T_number', bound=np.number[Any])
_T_int = TypeVar('_T_int', bound=np.integer[Any], default=np.intp)
_T_float = TypeVar('_T_float', bound=np.floating[Any], default=np.float64)

_T_size = TypeVar('_T_size', bound=int)

_T_shape0 = TypeVar('_T_shape0', bound=lnpt.AtLeast0D)
_T_shape1 = TypeVar('_T_shape1', bound=lnpt.AtLeast1D)
_T_shape2 = TypeVar('_T_shape2', bound=lnpt.AtLeast2D)

_DType: TypeAlias = np.dtype[_T_scalar] | type[_T_scalar]


def ensure_axis_at(
    a: npt.NDArray[_T_scalar],
    /,
    source: int | None,
    destination: int,
    *,
    order: lnpt.OrderReshape = 'C',
) -> npt.NDArray[_T_scalar]:
    """
    Moves the from `source` to `destination` if needed, or returns a flattened
    array is `source` is set to `None`.
    """
    if a.ndim <= 1 or source == destination:
        return a
    if source is None:
        return a.reshape(-1, order=order)

    source = source + a.ndim if source < 0 else source
    destination = destination + a.ndim if destination < 0 else destination

    return a if source == destination else np.moveaxis(a, source, destination)


def plotting_positions(
    n: int,
    /,
    alpha: float = 0.4,
    beta: float | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.float64]:
    """
    A re-implementation of [`scipy.stats.mstats.plotting_positions`
    ](scipy.stats.mstats.plotting_positions), but without the ridiculous
    interface.
    """
    x0 = 1 - alpha
    xn = x0 + n - (alpha if beta is None else beta)
    return np.linspace(x0 / xn, (x0 + n - 1) / xn, n, dtype=dtype)


def round0(
    a: lnpt.CanArray[_T_shape0, _T_float],
    /,
    tol: float | None = None,
) -> lnpt.Array[_T_shape0, _T_float]:
    """Replace all values `<= tol` with `0`."""
    _a = np.asarray(a)
    _tol = np.finfo(_a.dtype).resolution * 2 if tol is None else abs(tol)
    return np.where(np.abs(a) <= _tol, 0, a)


def _apply_aweights(
    x: lnpt.Array[_T_shape1, _T_float],
    v: lnpt.Array[_T_shape1 | lnpt.AtLeast1D, _T_float | lnpt.Float],
    axis: int,
) -> lnpt.Array[_T_shape1, _T_float]:
    # interpret the weights as horizontal coordinates using cumsum
    vv = np.cumsum(v, axis=axis)
    assert vv.shape == x.shape, (vv.shape, x.shape)

    # ensure that the samples are on the last axis, for easy iterating
    if swap_axes := axis % x.ndim != x.ndim - 1:
        x, vv = np.swapaxes(x, axis, -1), np.moveaxis(vv, axis, -1)

    # cannot use np.apply_along_axis here, since both x_k and w_k need to be
    # applied simultaneously
    out = np.empty_like(x)

    for j in np.ndindex(out.shape[:-1]):
        x_jk, w_jk = x[j], vv[j]
        if w_jk[-1] <= 0:
            msg = 'weight sum must be positive'
            raise ValueError(msg)

        # linearly interpolate to effectively "stretch" samples with large
        # weight, and "compress" those with small weights
        v_jk = np.linspace(w_jk[0], w_jk[-1], len(w_jk), dtype=np.float64)
        out[j] = np.interp(v_jk, w_jk, x_jk)

    # unswap the axes if previously swapped
    return np.swapaxes(out, -1, axis) if swap_axes else out


def _sort_like(
    a: lnpt.Array[_T_shape1, _T_number],
    i: lnpt.Array[tuple[int], np.integer[Any]],
    /,
    axis: int | None,
) -> lnpt.Array[_T_shape1, _T_number]:
    return (
        np.take(a, i, axis=None if a.ndim == i.ndim else axis)
        if min(a.ndim, i.ndim) <= 1
        else np.take_along_axis(a, i, axis)
    )


def sort_maybe(
    x: lnpt.Array[_T_shape1, _T_number],
    /,
    axis: int = -1,
    *,
    sort: bool | lnpt.SortKind = True,
    inplace: bool = False,
) -> lnpt.Array[_T_shape1, _T_number]:
    if not sort:
        return x

    kind = sort if isinstance(sort, str) else None

    if inplace:
        x.sort(axis=axis, kind=kind)
        return x
    return np.sort(x, axis=axis, kind=kind)


def ordered(  # noqa: C901
    x: lnpt.AnyArrayFloat,
    y: lnpt.AnyArrayFloat | None = None,
    /,
    axis: int | None = None,
    dtype: _DType[np.floating[Any]] | None = None,
    *,
    fweights: AnyFWeights | None = None,
    aweights: AnyAWeights | None = None,
    sort: lnpt.SortKind | bool = True,
) -> lnpt.Array[lnpt.AtLeast1D, lnpt.Float]:
    """
    Calculate `n = len(x)` order stats of `x`, optionally weighted.
    If `y` is provided, the order of `y` is used instead.
    """
    _x = _z = np.asanyarray(x, dtype=dtype)

    # ravel/flatten, without copying
    if axis is None:
        _x = _x.reshape(-1)

    # figure out the ordering
    if y is not None:
        _y = np.asanyarray(y)
        if axis is None:
            _y = _y.reshape(-1)

        #  sort first by y, then by x (faster than lexsort)
        if _y.ndim == _x.ndim:
            _z = _y + 1j * _x
        else:
            assert axis is not None
            _z = np.apply_along_axis(np.add, axis, 1j * _x, _y)

    # apply the ordering
    if sort or sort is None:  # pyright: ignore[reportUnnecessaryComparison]
        kind = sort if isinstance(sort, str) else None
        i_kk = np.argsort(_z, axis=axis, kind=kind)
        x_kk = _sort_like(_x, i_kk, axis=axis)
    else:
        if axis is None:
            i_kk = np.arange(len(_z))
        else:
            i_kk = np.mgrid[tuple(slice(0, j) for j in _z.shape)][axis]
        x_kk = _x

    # prepare observation weights
    w_kk = None
    if aweights is not None:
        w_kk = _sort_like(np.asanyarray(aweights), i_kk, axis=axis)

    # apply the frequency weights to x, and (optionally) to aweights
    if fweights is not None:
        r = np.asanyarray(fweights, int)
        r_kk = _sort_like(r, i_kk, axis=axis)

        # avoid unnecessary repeats by normalizing by the GCD
        if (gcd := np.gcd.reduce(r_kk)) <= 0:
            msg = 'fweights must be non-negative and have a positive sum'
            raise ValueError(msg)
        if gcd > 1:
            r_kk //= gcd

        if w_kk is not None:
            w_kk = np.repeat(w_kk, r_kk, axis=axis)

        x_kk = np.repeat(x_kk, r_kk, axis=axis)

    # optionally, apply the observation weights
    if w_kk is not None:
        x_kk = _apply_aweights(x_kk, w_kk, axis=axis or 0)

    return x_kk


def clean_order(
    r: AnyOrder,
    /,
    name: str = 'r',
    rmin: int = 0,
) -> int:
    """Validates and cleans an single (L-)moment order."""
    if (_r := int(r)) < rmin:
        msg = f'expected {name} >= {rmin}, got {_r}'
        raise TypeError(msg)

    return _r


def clean_orders(
    r: AnyOrderND,
    /,
    name: str = 'r',
    rmin: int = 0,
    dtype: _DType[_T_int] = np.intp,
) -> lnpt.Array[Any, _T_int]:
    """Validates and cleans an array-like of (L-)moment orders."""
    _r = np.asarray_chkfinite(r, dtype=dtype)

    if np.any(invalid := _r < rmin):
        i = np.argmax(invalid)
        msg = f'expected all {name} >= {rmin}, got {name}[{i}] = {_r[i]} '
        raise TypeError(msg)

    return _r


_COMMON_TRIM1: Final[frozenset[int]] = frozenset({0, 1, 2})
_COMMON_TRIM2: Final[frozenset[tuple[int, int]]] = frozenset(
    {(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (2, 0)},
)


def clean_trim(trim: AnyTrim, /) -> tuple[int, int] | tuple[float, float]:
    """
    Validates and cleans the passed trim; and return a 2-tuple of either ints
    or floats.

    Notes:
        - This uses `.is_integer()`, instead of `isinstance(int)`.
          So e.g. `clean_trim(1.0)` will return `tuple[int, int]`.
        - Although not allowed by typecheckers, numpy integer or floating
          scalars are also accepted, and will be converted to `int` or `float`.
    """
    # fast pass-through for the common cases
    if trim in _COMMON_TRIM1:
        return trim, trim
    if trim in _COMMON_TRIM2:
        return trim

    match trim:
        case s, t:
            pass
        case st:
            s = t = st

    fractional = False
    for f in map(float, (s, t)):
        if not math.isfinite(f):
            msg = 'trim orders must be finite'
            raise ValueError(msg)
        if f <= -1 / 2:
            msg = 'trim orders must be greater than -1/2'
            raise ValueError(msg)
        if not f.is_integer():
            fractional = True

    return (float(s), float(t)) if fractional else (int(s), int(t))


def moments_to_ratio(
    rs: lnpt.Array[Any, np.integer[Any]],
    l_rs: lnpt.Array[lnpt.AtLeast1D, _T_float],
    /,
) -> _T_float | npt.NDArray[_T_float]:
    """
    Using stacked order of shape (2, ...), and an L-moments array, returns
    the L-moment ratio's.
    """
    assert len(rs) == 2
    assert rs.shape[:l_rs.ndim] == l_rs.shape[:rs.ndim], [rs.shape, l_rs.shape]

    r_eq_s = rs[0] == rs[1]
    if r_eq_s.ndim < l_rs.ndim - 1:
        r_eq_s = r_eq_s.reshape(
            r_eq_s.shape + (1,) * (l_rs.ndim - r_eq_s.ndim - 1),
        )

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            r_eq_s,
            np.ones_like(l_rs[0]),
            np.divide(l_rs[0], l_rs[1], where=~r_eq_s),
        )[()]


def moments_to_stats_cov(
    t_0r: lnpt.Array[tuple[int], np.floating[Any]],
    ll_kr: lnpt.Array[_T_shape2, _T_float],
) -> lnpt.Array[_T_shape2, _T_float]:
    # t_0r are L-ratio's for r = 0, 1, ..., R (t_0r[0] == 1 / L-scale)
    # t_0r[1] isn't used, and can be set to anything
    # ll_kr is the L-moment cov of size R**2 (orders start at 1 here)
    assert len(t_0r) > 0
    assert (len(t_0r) - 1)**2 == ll_kr.size

    t_0, t_r = t_0r[0], t_0r[1:]
    tt_kr = np.empty_like(ll_kr)
    for k, r in zip(*np.triu_indices_from(tt_kr), strict=True):
        if r <= 1:
            tt = ll_kr[k, r]
        elif k <= 1:
            tt = t_0 * (ll_kr[k, r] - ll_kr[1, k] * t_r[r])
        else:
            tt = t_0**2 * (
                ll_kr[k, r]
                - ll_kr[1, k] * t_r[r]
                - ll_kr[1, r] * t_r[k]
                + ll_kr[1, 1] * t_r[k] * t_r[r]
            )

        tt_kr[k, r] = tt_kr[r, k] = tt

    return tt_kr


def l_stats_orders(
    num: _T_size,
    /,
    dtype: _DType[_T_int] = np.intp,
) -> tuple[
    lnpt.Array[tuple[_T_size], _T_int],
    lnpt.Array[tuple[_T_size], _T_int],
]:
    """
    Create the L-moment order array `[1, 2, ..., r]` and corresponding
    ratio array `[0, 0, 2, ...]` of same size.
    """
    r = np.arange(1, num + 1, dtype=dtype)
    s = np.array([0] * min(2, num) + [2] * (num - 2), dtype=dtype)
    return r, s
