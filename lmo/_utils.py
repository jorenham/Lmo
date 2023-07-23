__all__ = (
    'clean_order',
    'ensure_axis_at',
    'as_float_array',
    'ordered',
    'moments_to_ratio',
)

from typing import Any, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from .typing import IndexOrder, IntVector, SortKind

T = TypeVar('T', bound=np.generic)
FT = TypeVar('FT', bound=np.floating[Any])


def clean_order(
    order: SupportsIndex,
    /,
    name: str = 'r',
    strict: bool = False,
) -> int:
    if (r := order.__index__()) < (r0 := int(strict)):
        msg = f'expected {name} >= {r0}, got {r}'
        raise TypeError(msg)

    return r


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
    out = x if isinstance(x.dtype.type, np.floating) else x.astype(np.float_)

    # the `_[()]` ensures that 0-d arrays become scalars
    return (out.ravel() if flat and out.ndim != 1 else out)[()]


def _apply_aweights(
    x: npt.NDArray[np.floating[Any]],
    v: npt.NDArray[np.floating[Any]],
    axis: int,
) -> npt.NDArray[np.float_]:
    # interpret the weights as horizontal coordinates using cumsum
    vv = np.cumsum(v, axis=axis)
    assert vv.shape == x.shape, (vv.shape, x.shape)

    # ensure that the samples are on the last axis, for easy iterating
    if swap_axes := axis % x.ndim != x.ndim - 1:
        x, vv = np.swapaxes(x, axis, -1), np.moveaxis(vv, axis, -1)

    # cannot use np.apply_along_axis here, since both x_k and w_k need to be
    # applied simultaneously
    out = np.empty(x.shape, dtype=np.float_)

    x_jk: npt.NDArray[np.floating[Any]]
    w_jk: npt.NDArray[np.floating[Any]]
    v_jk: npt.NDArray[np.float_]
    for j in np.ndindex(out.shape[:-1]):
        x_jk, w_jk = x[j], vv[j]
        if w_jk[-1] <= 0:
            msg = 'weight sum must be positive'
            raise ValueError(msg)

        # linearly interpolate to effectively "stretch" samples with large
        # weight, and "compress" those with small weights
        v_jk = np.linspace(w_jk[0], w_jk[-1], len(w_jk), dtype=np.float_)
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
    sort: SortKind | None = 'stable',
) -> npt.NDArray[np.floating[Any]]:
    """
    Calculate `n = len(x)` order stats of `x`, optionally weighted.
    If `y` is provided, the order of `y` is used instead.
    """
    if fweights is not None:
        # avoid uneccesary repeats by normalizing by the GCD
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
        return np.sort(_x, axis=axis, kind=sort)
    if y is not None:
        _y = _clean_array(y)
        i_k = np.argsort(_y, axis=axis if _y.ndim > 1 else -1, kind=sort)
    else:
        i_k = np.argsort(_x, axis=axis, kind=sort)

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


def moments_to_ratio(
    rs: npt.NDArray[np.integer[Any]],
    l_rs: npt.NDArray[FT],
    /,
) -> FT | npt.NDArray[FT]:
    assert rs.shape == l_rs.shape, [rs.shape, l_rs.shape]
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
