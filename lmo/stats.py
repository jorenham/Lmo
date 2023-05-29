__all__ = 'order_stats', 'l_ratio_max'

from math import factorial as fact
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from ._utils import as_float_array, clean_order
from .typing import IntVector, SortKind


T = TypeVar('T', bound=npt.NBitBase)


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
            raise ValueError('weight sum must be positive')

        # linearly interpolate to effectively "stretch" samples with large
        # weight, and "compress" those with small weights
        v_jk = np.linspace(w_jk[0], w_jk[-1], len(w_jk), dtype=np.float_)
        out[j] = np.interp(v_jk, w_jk, x_jk)  # pyright: ignore

    # unswap the axes if previously swapped
    return np.swapaxes(out, -1, axis) if swap_axes else out


def order_stats(
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
            raise ValueError(
                'fweights must be non-negative and have a positive sum'
            )
        r = r // gcd if gcd > 1 else r
    else:
        r = None

    def _clean_array(a: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        out = as_float_array(a, dtype=dtype, flat=axis is None)
        return out if r is None else np.repeat(out, r, axis=axis)

    _x = _clean_array(x)

    if aweights is None and y is None:
        return np.sort(_x, axis=axis, kind=sort)
    elif y is not None:
        _y = _clean_array(y)
        i_k = np.argsort(_y, axis=axis if _y.ndim > 1 else -1, kind=sort)
    else:
        i_k = np.argsort(_x, axis=axis, kind=sort)

    def _sort_like(
        a: npt.NDArray[np.floating[T]],
    ) -> npt.NDArray[np.floating[T]]:
        return (
            np.take(  # pyright: ignore [reportUnknownMemberType]
                a,
                i_k,
                axis=None if a.ndim == i_k.ndim else axis
            )
            if min(a.ndim, i_k.ndim) <= 1
            else np.take_along_axis(a, i_k, axis)
        )

    x_k = _sort_like(_x)

    if aweights is None:
        return x_k

    w_k = _sort_like(_clean_array(aweights))
    return _apply_aweights(x_k, w_k, axis=axis or 0)


def l_ratio_max(
    r: int,
    s: int = 2,
    /,
    trim: tuple[int, int] = (0, 0),
) -> float:
    """
    The theoretical upper bound on the absolute TL-ratios, i.e.::

        abs(lmo.l_ratio(a, r, s, trim)) <= tl_ratio_max(r, s, trim)

    is True for all samples `a`.

    References:
        * Hosking, J.R.M., Some Theory and Practical Uses of Trimmed L-moments.
          Page 6, equation 14.

    """

    # the zeroth (TL-)moment is 1. I.e. the total area under the pdf (or the
    # sum of the ppf if discrete) is 1.
    _r = clean_order(r)
    _s = clean_order(s, name='s')

    if _r in (0, _s):
        return 1.0
    if not _s:
        return float('inf')

    t1, t2 = trim
    m = min(t1, t2)

    # disclaimer: the `k` instead of a `2` here is just a guess
    return (
        _s * fact(m + _s - 1) * fact(t1 + t2 + _r) /
        (_r * fact(m + _r - 1) * fact(t1 + t2 + _s))
    )
