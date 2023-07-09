__all__ = 'ordered', 'ostat_from_ppf', 'l_moment_from_ppf'

import functools
import math
from typing import Any, Callable, TypeVar, cast

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.special

from ._utils import as_float_array
from .typing import AnyInt, IntVector, SortKind

T = TypeVar('T', bound=np.floating[Any])


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

    def _sort_like(a: npt.NDArray[T]) -> npt.NDArray[T]:
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



R = TypeVar('R', bound=float | npt.NDArray[np.float_])


def ostat_from_ppf(
    ppf: Callable[[float], R],
    /,
    *,
    p_min: float = 0.0,
    p_max: float = 1.0,
    cache: bool = True,
) -> Callable[[AnyInt, AnyInt], R]:
    # TODO: docstring, example, tests

    ppf_cached: Callable[[float], R] = functools.cache(ppf) if cache else ppf

    def u_ppf(
        p: float,
        i: int,
        n: int,
        /,
    ) -> R:
        return cast(
            R,
            ppf_cached(p)
            * p**(i - 1)
            * (1 - p)**(n - i)
            * math.comb(n - 1, i - 1)
            * n
        )

    def u_mean(i: AnyInt, n: AnyInt, /) -> R:
        if not (0 < i <= n):
            raise ValueError(f'expected 0 < i <= n, got {i = } and {n = }')

        return scipy.integrate.quad( # type: ignore
            u_ppf,
            p_min,
            p_max,
            args=(int(i), int(n)),
        )[0]

    return u_mean


def l_moment_from_ppf(
    ppf: Callable[[float], R],
    r: AnyInt | IntVector,
    /,
    trim: tuple[int, int] = (0, 0),
    **options: Any,
) -> float | npt.NDArray[np.float_]:
    # TODO: docstring, example, tests

    t1, t2 = trim
    if t1 < 0 or t2 < 0:
        raise ValueError('trim values must be non-negative')

    u = ostat_from_ppf(ppf, **options)

    def l_moment(_r: AnyInt) -> float | npt.NDArray[np.float_]:
        if _r < 0:
            raise ValueError(f'r must be >=0, got {_r}')
        if _r == 0:
            return 1.0

        return np.array([
            (-1) ** _k
            * math.comb(_r - 1, _k)
            * u(t1 + _r - _k, t1 + t2 + _r)
            for _k in range(int(_r))
        ]).mean()

    rs = np.asarray(r, np.int_).ravel()
    if rs.size == 0:
        return l_moment(rs[0])

    l_r = np.empty(rs.shape)
    for ix in np.ndindex(*rs.shape):
        l_r[ix] = l_moment(rs[ix])

    return np.round(l_r, 12)
