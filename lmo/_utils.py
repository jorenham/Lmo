__all__ = (
    'as_float_array',
    'broadstack',
    'ensure_axis_at',
    'plotting_positions',
    'round0',
    'ordered',

    'clean_order',
    'clean_orders',
    'clean_trim',

    'moments_to_ratio',
    'moments_to_stats_cov',
    'l_stats_orders',
)

from typing import Any, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from .typing import AnyInt, AnyTrim, IndexOrder, IntVector, SortKind

T = TypeVar('T', bound=np.generic)
FT = TypeVar('FT', bound=np.floating[Any])


def as_float_array(
    a: npt.ArrayLike,
    /,
    dtype: npt.DTypeLike = None,
    order: IndexOrder | None = None,
    *,
    check_finite: bool = False,
    flat: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """
    Convert to array if needed, and only cast to float64 dtype if not a
    floating type already. Similar as in e.g. `numpy.mean`.
    """
    asarray = np.asarray_chkfinite if check_finite else np.asarray

    x = asarray(a, dtype=dtype, order=order)
    out = x if isinstance(x.dtype.type, np.floating) else x.astype(np.float64)

    # the `_[()]` ensures that 0-d arrays become scalars
    return (out.ravel() if flat and out.ndim != 1 else out)[()]


def broadstack(
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
) -> npt.NDArray[np.int_]:
    return np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))


def ensure_axis_at(
    a: npt.NDArray[T],
    /,
    source: int | None,
    destination: int,
    order: IndexOrder = 'C',
) -> npt.NDArray[T]:
    if a.ndim <= 1 or source == destination:
        return a
    if source is None:
        return a.ravel(order)

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


def round0(a: npt.NDArray[T], /, tol: float = 1e-8) -> npt.NDArray[T]:
    """Round values close to zero."""
    return np.where(np.abs(a) <= abs(tol), 0, a) if tol else a


def _apply_aweights(
    x: npt.NDArray[np.floating[Any]],
    v: npt.NDArray[np.floating[Any]],
    axis: int,
) -> npt.NDArray[np.float64]:
    # interpret the weights as horizontal coordinates using cumsum
    vv = np.cumsum(v, axis=axis)
    assert vv.shape == x.shape, (vv.shape, x.shape)

    # ensure that the samples are on the last axis, for easy iterating
    if swap_axes := axis % x.ndim != x.ndim - 1:
        x, vv = np.swapaxes(x, axis, -1), np.moveaxis(vv, axis, -1)

    # cannot use np.apply_along_axis here, since both x_k and w_k need to be
    # applied simultaneously
    out = np.empty(x.shape, dtype=np.float64)

    x_jk: npt.NDArray[np.floating[Any]]
    w_jk: npt.NDArray[np.floating[Any]]
    v_jk: npt.NDArray[np.float64]
    for j in np.ndindex(out.shape[:-1]):
        x_jk, w_jk = x[j], vv[j]
        if w_jk[-1] <= 0:
            msg = 'weight sum must be positive'
            raise ValueError(msg)

        # linearly interpolate to effectively "stretch" samples with large
        # weight, and "compress" those with small weights
        v_jk = np.linspace(w_jk[0], w_jk[-1], len(w_jk), dtype=np.float64)
        out[j] = np.interp(v_jk, w_jk, x_jk)  # pyright: ignore

    # unswap the axes if previously swapped
    return np.swapaxes(out, -1, axis) if swap_axes else out


def ordered(
    x: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    /,
    axis: int | None = None,
    dtype: npt.DTypeLike = None,
    *,
    fweights: IntVector | None = None,
    aweights: npt.ArrayLike | None = None,
    sort: SortKind | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Calculate `n = len(x)` order stats of `x`, optionally weighted.
    If `y` is provided, the order of `y` is used instead.
    """
    if fweights is not None:
        # avoid unnecessary repeats by normalizing by the GCD
        r = np.asarray(fweights)
        # noinspection PyUnresolvedReferences
        if (gcd := np.gcd.reduce(r)) <= 0:
            msg = 'fweights must be non-negative and have a positive sum'
            raise ValueError(msg)

        r = r // gcd if gcd > 1 else r
    else:
        r = None

    def _clean_array(a: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        out = as_float_array(a, dtype=dtype, flat=axis is None)
        return out if r is None else np.repeat(out, r, axis=axis)

    _x = _clean_array(x)

    if aweights is None and y is None:
        return np.sort(_x, axis=axis, kind=sort)  # type: ignore
    if y is not None:
        _y = _clean_array(y)
        i_k = np.argsort(_y, axis=axis if _y.ndim > 1 else -1, kind=sort)
    else:
        i_k = np.argsort(_x, axis=axis, kind=sort)  # type: ignore

    def _sort_like(a: npt.NDArray[T]) -> npt.NDArray[T]:
        return (
            np.take(  # pyright: ignore [reportUnknownMemberType]
                a,
                i_k,
                axis=None if a.ndim == i_k.ndim else axis,
            )
            if min(a.ndim, i_k.ndim) <= 1
            else np.take_along_axis(a, i_k, axis)
        )

    x_k = _sort_like(_x)

    if aweights is None:
        return x_k

    w_k = _sort_like(_clean_array(aweights))
    return _apply_aweights(x_k, w_k, axis=axis or 0)


def clean_order(
    r: SupportsIndex,
    /,
    name: str = 'r',
    rmin: int = 0,
) -> int:
    if (_r := r.__index__()) < rmin:
        msg = f'expected {name} >= {rmin}, got {_r}'
        raise TypeError(msg)

    return _r


def clean_orders(
    r: IntVector | AnyInt,
    /,
    name: str = 'r',
    rmin: int = 0,
) -> npt.NDArray[np.int_]:
    _r = np.asarray_chkfinite(r, np.int_)

    if np.any(invalid := _r < rmin):
        i = np.argmax(invalid)
        msg = f'expected all {name} >= {rmin}, got {name}[{i}] = {_r[i]} '
        raise TypeError(msg)

    return _r


def clean_trim(trim: AnyTrim) -> tuple[int, int] | tuple[float, float]:
    _trim = np.asarray_chkfinite(trim)

    if not np.isrealobj(_trim):
        msg = 'trim must be real'
        raise TypeError(msg)

    if _trim.ndim > 1:
        msg = 'trim cannot be vectorized'
        raise TypeError(trim)

    n = _trim.size
    if n == 0:
        _trim = np.array([0, 0])
    if n == 1:
        _trim = np.repeat(_trim, 2)
    elif n > 2:
        msg = f'expected two trim values, got {n} instead'
        raise TypeError(msg)

    s, t = _trim

    if s <= -1/2 or t <= -1/2:
        msg = f'trim must both be >-1/2, got {(s, t)}'
        raise ValueError(msg)

    if s.is_integer() and t.is_integer():
        return int(s), int(t)

    return float(s), float(t)


def moments_to_ratio(
    rs: npt.NDArray[np.integer[Any]],
    l_rs: npt.NDArray[FT],
    /,
) -> FT | npt.NDArray[FT]:
    assert rs.shape[:l_rs.ndim] == l_rs.shape[:rs.ndim], [rs.shape, l_rs.shape]
    assert len(rs) == 2

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
    t_0r: npt.NDArray[np.float64],
    ll_kr: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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
    num: int,
    /,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    return (
        np.arange(1, num + 1),
        np.array([0] * min(2, num) + [2] * (num - 2)),
    )
