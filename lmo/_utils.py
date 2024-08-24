from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast, overload

import numpy as np
import numpy.typing as npt


if sys.version_info >= (3, 13):
    from typing import LiteralString, TypeVar
else:
    from typing_extensions import LiteralString, TypeVar

if TYPE_CHECKING:
    import optype.numpy as onpt

    import lmo.typing as lmt
    import lmo.typing.np as lnpt

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


_SCT = TypeVar('_SCT', bound=np.generic)
_SCT_uifc = TypeVar('_SCT_uifc', bound='lnpt.Number')
_SCT_ui = TypeVar('_SCT_ui', bound='lnpt.Int', default=np.int_)
_SCT_f = TypeVar('_SCT_f', bound='lnpt.Float', default=np.float64)

_DT_f = TypeVar('_DT_f', bound=np.dtype['lnpt.Float'])
_AT_f = TypeVar('_AT_f', bound='npt.NDArray[lnpt.Float] | lnpt.Float')

_SizeT = TypeVar('_SizeT', bound=int)

_ShapeT = TypeVar('_ShapeT', bound='onpt.AtLeast0D')
_ShapeT1 = TypeVar('_ShapeT1', bound='onpt.AtLeast1D')
_ShapeT2 = TypeVar('_ShapeT2', bound='onpt.AtLeast2D')

_DType: TypeAlias = np.dtype[_SCT] | type[_SCT]


def ensure_axis_at(
    a: npt.NDArray[_SCT],
    /,
    source: int | None,
    destination: int,
    *,
    order: lnpt.OrderReshape = 'C',
) -> npt.NDArray[_SCT]:
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
    n: _SizeT,
    /,
    alpha: float = 0.4,
    beta: float | None = None,
) -> onpt.Array[tuple[_SizeT], np.float64]:
    """
    A re-implementation of [`scipy.stats.mstats.plotting_positions`
    ](scipy.stats.mstats.plotting_positions), but without the ridiculous
    interface.
    """
    x0 = 1 - alpha
    xn = x0 + n - (alpha if beta is None else beta)
    return np.linspace(x0 / xn, (x0 + n - 1) / xn, n, dtype=np.float64)


@overload
def round0(a: _AT_f, /, tol: float | None = ...) -> _AT_f: ...
@overload
def round0(
    a: onpt.CanArray[_ShapeT, _DT_f],
    /,
    tol: float | None = ...,
) -> np.ndarray[_ShapeT, _DT_f]: ...
@overload
def round0(a: float, /, tol: float | None = ...) -> np.float64: ...
def round0(
    a: float | onpt.CanArray[_ShapeT, np.dtype[_SCT_f]],
    /,
    tol: float | None = None,
) -> onpt.Array[_ShapeT, _SCT_f] | _SCT_f:
    """
    Replace all values `<= tol` with `0`.

    Todo:
        - Add an `inplace: bool = False` kwarg
    """
    _a = np.asanyarray(a)
    _tol = np.finfo(_a.dtype).resolution * 2 if tol is None else abs(tol)
    out = np.where(np.abs(_a) <= _tol, 0, a)
    return out[()] if np.isscalar(a) else out


def _apply_aweights(
    x: np.ndarray[_ShapeT1, _DT_f],
    v: np.ndarray[_ShapeT1 | onpt.AtLeast1D, _DT_f | np.dtype[lnpt.Float]],
    axis: int,
) -> np.ndarray[_ShapeT1, _DT_f]:
    # interpret the weights as horizontal coordinates using cumsum
    vv = np.cumsum(v, axis=axis)
    assert vv.shape == x.shape, (vv.shape, x.shape)

    # ensure that the samples are on the last axis, for easy iterating
    if swap_axes := axis % x.ndim != x.ndim - 1:
        x = cast(np.ndarray[_ShapeT1, _DT_f], np.swapaxes(x, axis, -1))
        vv = np.moveaxis(vv, axis, -1)

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
    if swap_axes:
        out = cast(np.ndarray[_ShapeT1, _DT_f], np.swapaxes(out, -1, axis))

    return out


def _sort_like(
    a: onpt.Array[_ShapeT1, _SCT_uifc],
    i: onpt.Array[tuple[int], lnpt.Int],
    /,
    axis: int | None,
) -> onpt.Array[_ShapeT1, _SCT_uifc]:
    return (
        np.take(a, i, axis=None if a.ndim == i.ndim else axis)
        if min(a.ndim, i.ndim) <= 1
        else np.take_along_axis(a, i, axis)
    )


def sort_maybe(
    x: onpt.Array[_ShapeT1, _SCT_uifc],
    /,
    axis: int = -1,
    *,
    sort: bool | lnpt.SortKind = True,
    inplace: bool = False,
) -> onpt.Array[_ShapeT1, _SCT_uifc]:
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
    dtype: _DType[lnpt.Float] | None = None,
    *,
    fweights: lmt.AnyFWeights | None = None,
    aweights: lmt.AnyAWeights | None = None,
    sort: lnpt.SortKind | bool = True,
) -> onpt.Array[onpt.AtLeast1D, lnpt.Float]:
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
    r: lmt.AnyOrder,
    /,
    name: LiteralString = 'r',
    rmin: int = 0,
) -> int:
    """Validates and cleans an single (L-)moment order."""
    if (_r := int(r)) < rmin:
        msg = f'expected {name} >= {rmin}, got {_r}'
        raise TypeError(msg)

    return _r


def clean_orders(
    r: lmt.AnyOrderND,
    /,
    name: str = 'r',
    rmin: int = 0,
    dtype: _DType[_SCT_ui] = np.int_,
) -> onpt.Array[Any, _SCT_ui]:
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


@overload
def clean_trim(trim: lmt.AnyTrimInt, /) -> tuple[int, int]: ...
@overload
def clean_trim(trim: lmt.AnyTrimFloat, /) -> tuple[float, float]: ...
def clean_trim(trim: lmt.AnyTrim, /) -> tuple[int, int] | tuple[float, float]:
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
    rs: onpt.Array[tuple[int, ...], lnpt.Int],
    l_rs: onpt.Array[onpt.AtLeast1D, _SCT_f],
    /,
) -> _SCT_f | npt.NDArray[_SCT_f]:
    """
    Using stacked order of shape (2, ...), and an L-moments array, returns
    the L-moment ratio's.
    """
    assert len(rs) == 2
    assert rs.shape[: l_rs.ndim] == l_rs.shape[: rs.ndim], [
        rs.shape,
        l_rs.shape,
    ]

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
    t_0r: onpt.Array[tuple[int], lnpt.Float],
    ll_kr: onpt.Array[_ShapeT2, _SCT_f],
) -> onpt.Array[_ShapeT2, _SCT_f]:
    # t_0r are L-ratio's for r = 0, 1, ..., R (t_0r[0] == 1 / L-scale)
    # t_0r[1] isn't used, and can be set to anything
    # ll_kr is the L-moment cov of size R**2 (orders start at 1 here)
    assert len(t_0r) > 0
    assert (len(t_0r) - 1) ** 2 == ll_kr.size

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
    num: _SizeT,
    /,
    dtype: _DType[_SCT_ui] = np.int_,
) -> tuple[
    onpt.Array[tuple[_SizeT], _SCT_ui],
    onpt.Array[tuple[_SizeT], _SCT_ui],
]:
    """
    Create the L-moment order array `[1, 2, ..., r]` and corresponding
    ratio array `[0, 0, 2, ...]` of same size.
    """
    r = np.arange(1, num + 1, dtype=dtype)
    s = np.array([0] * min(2, num) + [2] * (num - 2), dtype=dtype)
    return r, s
