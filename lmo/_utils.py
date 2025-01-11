from __future__ import annotations

import math
from typing import Final, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc
import optype.typing as opt

import lmo.typing as lmt
from .errors import InvalidLMomentError
from .special import fpow

__all__ = (
    "clean_order",
    "clean_orders",
    "clean_trim",
    "ensure_axis_at",
    "l_stats_orders",
    "moments_to_ratio",
    "moments_to_stats_cov",
    "ordered",
    "plotting_positions",
    "round0",
    "sort_maybe",
    "transform_moments",
    "validate_moments",
)

###

_IntND: TypeAlias = onp.ArrayND[npc.integer]
_FloatND: TypeAlias = onp.ArrayND[npc.floating]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=np.generic)
_FloatT = TypeVar("_FloatT", bound=npc.floating)
_NumberT = TypeVar("_NumberT", bound=npc.number)
_FloatNDT = TypeVar("_FloatNDT", bound=_FloatND)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_Tuple2: TypeAlias = tuple[_T, _T]
_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]

###


def ensure_axis_at(
    a: onp.ArrayND[_SCT],
    /,
    source: op.CanIndex | None,
    destination: op.CanIndex,
    *,
    order: lmt.OrderReshape = "C",
    copy: bool | None = None,
) -> onp.ArrayND[_SCT]:
    """
    Moves the from `source` to `destination` if needed, or returns a flattened
    array is `source` is set to `None`.
    """
    if a.ndim <= 1:
        return a.copy() if copy else a

    if source is None:
        return a.reshape(-1, order=order, copy=copy)

    if (src := int(source)) == (dst := int(destination)):
        return a.copy() if copy else a

    if src < 0:
        src += a.ndim
    if dst < 0:
        dst += a.ndim

    return np.moveaxis(a, src, dst)


def plotting_positions(
    n: onp.ToInt,
    /,
    alpha: onp.ToFloat = 0.4,
    beta: onp.ToFloat | None = None,
) -> _FloatND:
    """
    A re-implementation of [`scipy.stats.mstats.plotting_positions`
    ](scipy.stats.mstats.plotting_positions), but without the ridiculous
    interface.
    """
    x0 = 1 - alpha
    xn = x0 + n - (alpha if beta is None else beta)
    return np.linspace(x0 / xn, (x0 + n - 1) / xn, n)


@overload
def round0(a: onp.ToFloat, /, tol: float | None = None) -> float: ...
@overload
def round0(a: _FloatNDT, /, tol: float | None = None) -> _FloatNDT: ...
@overload
def round0(a: onp.ToFloatND, /, tol: float | None = None) -> _FloatND: ...
def round0(
    a: onp.ToFloat | onp.ToFloatND,
    /,
    tol: float | None = None,
) -> float | _FloatND:
    """
    Replace all values `<= tol` with `0`.

    Todo:
        - Add an `inplace: bool = False` kwarg
    """
    a_ = np.asanyarray(a)
    if a_.dtype.kind not in "fc":
        return a_.item() if a_.ndim == 0 and np.isscalar(a) else a_

    tol_ = np.finfo(a_.dtype).resolution * 2 if tol is None else abs(tol)
    out = np.where(np.abs(a_) <= tol_, 0, a_)
    return out[()] if np.isscalar(a) else out


def _apply_aweights(x: _FloatNDT, v: _FloatND, axis: op.CanIndex) -> _FloatNDT:
    # interpret the weights as horizontal coordinates using cumsum
    vv = np.cumsum(v, axis=axis)
    assert vv.shape == x.shape, (vv.shape, x.shape)

    # ensure that the samples are on the last axis, for easy iterating
    axis = int(axis)
    if swap_axes := axis % x.ndim != x.ndim - 1:
        x_ = np.swapaxes(x, axis, -1)
        vv_ = np.moveaxis(vv, axis, -1)
    else:
        x_, vv_ = x, vv

    # cannot use np.apply_along_axis here, since both x_k and w_k need to be
    # applied simultaneously
    out: _FloatNDT = np.empty_like(x)

    for j in np.ndindex(out.shape[:-1]):
        x_jk, w_jk = x_[j], vv_[j]
        if w_jk[-1] <= 0:
            msg = "weight sum must be positive"
            raise ValueError(msg)

        # linearly interpolate to effectively "stretch" samples with large
        # weight, and "compress" those with small weights
        v_jk = np.linspace(w_jk[0], w_jk[-1], len(w_jk), dtype=np.float64)
        out[j] = np.interp(v_jk, w_jk, x_jk)

    # unswap the axes if previously swapped
    if swap_axes:
        out = np.swapaxes(out, -1, axis)  # pyright: ignore[reportAssignmentType]

    return out


def _sort_like(
    a: onp.ArrayND[_NumberT],
    i: _IntND,
    /,
    axis: op.CanIndex | None,
) -> onp.ArrayND[_NumberT]:
    return (
        np.take(a, i, axis=None if a.ndim == i.ndim else axis)
        if a.ndim <= 1 or i.ndim <= 1
        else np.take_along_axis(a, i, None if axis is None else int(axis))
    )


def sort_maybe(
    x: onp.ArrayND[_NumberT],
    /,
    axis: op.CanIndex = -1,
    *,
    sort: bool | lmt.SortKind = True,
    inplace: bool = False,
) -> onp.ArrayND[_NumberT]:
    if not sort:
        return x

    kind = sort if isinstance(sort, str) else None

    if inplace:
        x.sort(axis=axis, kind=kind)
        return x
    return np.sort(x, axis=axis, kind=kind)


@overload
def ordered(
    x: onp.ToFloatND,
    y: onp.ToFloatND | None = None,
    /,
    axis: op.CanIndex | None = None,
    dtype: None = None,
    *,
    fweights: lmt.ToFWeights | None = None,
    aweights: lmt.ToAWeights | None = None,
    sort: lmt.SortKind | bool = True,
) -> _FloatND: ...
@overload
def ordered(
    x: onp.ToFloatND,
    y: onp.ToFloatND | None,
    /,
    axis: op.CanIndex | None,
    dtype: _ToDType[_FloatT],
    *,
    fweights: lmt.ToFWeights | None = None,
    aweights: lmt.ToAWeights | None = None,
    sort: lmt.SortKind | bool = True,
) -> onp.ArrayND[_FloatT]: ...
@overload
def ordered(
    x: onp.ToFloatND,
    y: onp.ToFloatND | None = None,
    /,
    axis: op.CanIndex | None = None,
    *,
    dtype: _ToDType[_FloatT],
    fweights: lmt.ToFWeights | None = None,
    aweights: lmt.ToAWeights | None = None,
    sort: lmt.SortKind | bool = True,
) -> _FloatND: ...
def ordered(  # noqa: C901
    x: onp.ToFloatND,
    y: onp.ToFloatND | None = None,
    /,
    axis: op.CanIndex | None = None,
    dtype: onp.AnyFloatingDType | None = None,
    *,
    fweights: lmt.ToFWeights | None = None,
    aweights: lmt.ToAWeights | None = None,
    sort: lmt.SortKind | bool = True,
) -> _FloatND:
    """
    Calculate `n = len(x)` order stats of `x`, optionally weighted.
    If `y` is provided, the order of `y` is used instead.
    """
    x_ = z = np.asanyarray(x, dtype=dtype)

    # ravel/flatten, without copying
    if axis is None:
        x_ = x_.reshape(-1)

    # figure out the ordering
    if y is not None:
        y_ = np.asanyarray(y)
        if axis is None:
            y_ = y_.reshape(-1)

        #  sort first by y, then by x (faster than lexsort)
        if y_.ndim == x_.ndim:
            z = y_ + 1j * x_
        else:
            assert axis is not None
            z = np.apply_along_axis(np.add, axis, 1j * x_, y_)

    # apply the ordering
    if sort or sort is None:  # pyright: ignore[reportUnnecessaryComparison]
        kind = sort if isinstance(sort, str) else None
        i_kk = np.argsort(z, axis=axis, kind=kind)
        x_kk = _sort_like(x_, i_kk, axis=axis)
    else:
        if axis is None:
            i_kk = np.arange(len(z))
        else:
            i_kk = np.mgrid[tuple(slice(0, j) for j in z.shape)][axis]
        x_kk = x_

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
            msg = "fweights must be non-negative and have a positive sum"
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


@overload
def clean_order(r: lmt.ToOrder0D, /, name: str = "r", rmin: onp.ToInt = 0) -> int: ...
@overload
def clean_order(
    r: lmt.ToOrderND,
    /,
    name: str = "r",
    rmin: onp.ToInt = 0,
) -> onp.ArrayND[np.intp]: ...
def clean_order(
    r: lmt.ToOrder,
    /,
    name: str = "r",
    rmin: onp.ToInt = 0,
) -> int | onp.ArrayND[np.intp]:
    """Validates and cleans an single (L-)moment order."""
    if not isinstance(r, int | np.integer):
        return clean_orders(r, name=name, rmin=rmin)

    if r < rmin:
        msg = f"expected {name} >= {rmin}, got {r}"
        raise TypeError(msg)

    return int(r)


@overload
def clean_orders(
    r: lmt.ToOrder1D,
    /,
    name: str = "r",
    rmin: onp.ToInt = 0,
) -> onp.Array1D[np.intp]: ...
@overload
def clean_orders(
    r: lmt.ToOrderND,
    /,
    name: str = "r",
    rmin: onp.ToInt = 0,
) -> onp.ArrayND[np.intp]: ...
def clean_orders(
    r: lmt.ToOrderND,
    /,
    name: str = "r",
    rmin: onp.ToInt = 0,
) -> onp.ArrayND[np.intp]:
    """Validates and cleans an array-like of (L-)moment orders."""
    r_ = np.asarray_chkfinite(r, dtype=np.intp)

    if np.any(invalid := r_ < rmin):
        i = np.argmax(invalid)
        msg = f"expected all {name} >= {rmin}, got {name}[{i}] = {r_[i]} "
        raise TypeError(msg)

    return r_


_COMMON_TRIM1: Final[frozenset[int]] = frozenset({0, 1, 2})
_COMMON_TRIM2: Final[frozenset[tuple[int, int]]] = frozenset(
    {(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (2, 0)},
)


@overload
def clean_trim(trim: lmt.ToIntTrim, /) -> tuple[int, int]: ...
@overload
def clean_trim(trim: lmt.ToTrim, /) -> tuple[float, float]: ...
def clean_trim(trim: lmt.ToTrim, /) -> tuple[float, float]:
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
            msg = "trim orders must be finite"
            raise ValueError(msg)
        if f <= -1 / 2:
            msg = "trim orders must be greater than -1/2"
            raise ValueError(msg)
        if not f.is_integer():
            fractional = True

    return (float(s), float(t)) if fractional else (int(s), int(t))


def moments_to_ratio(
    rs: _IntND,
    l_rs: onp.ArrayND[_FloatT],
    /,
) -> float | onp.ArrayND[_FloatT]:
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

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            r_eq_s,
            np.ones_like(l_rs[0]),
            np.divide(l_rs[0], l_rs[1], where=~r_eq_s),
        )

    return out.item() if out.ndim == 0 else out


def moments_to_stats_cov(
    t_0r: _FloatND,
    ll_kr: onp.Array[_ShapeT, _FloatT],
) -> onp.Array[_ShapeT, _FloatT]:
    # t_0r are L-ratio's for r = 0, 1, ..., R (t_0r[0] == 1 / L-scale)
    # t_0r[1] isn't used, and can be set to anything
    # ll_kr is the L-moment cov of size R**2 (orders start at 1 here)
    assert t_0r.size
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


def l_stats_orders(num: opt.AnyInt, /) -> _Tuple2[onp.ArrayND[np.int32]]:
    """
    Create the L-moment order array `[1, 2, ..., r]` and corresponding
    ratio array `[0, 0, 2, ...]` of same size.
    """
    n = int(num)
    return (
        np.arange(1, n + 1, dtype=np.int32),
        np.array([0] * min(2, n) + [2] * (n - 2), dtype=np.int32),
    )


def transform_moments(
    r: onp.ArrayND[npc.integer],
    l_r: onp.Array[_ShapeT, _FloatT],
    /,
    shift: float = 0.0,
    scale: float = 1.0,
) -> None:
    """Apply a linear transformation to the L-moments in-place."""
    if scale != 1:
        l_r[r > 0] *= scale
    if shift != 0:
        l_r[r == 1] += shift


def validate_moments(l_r: _FloatND, s: float, t: float) -> None:
    """Vaalidate the L-moments with order [1, 2, 3, ...]."""
    if (l2 := l_r[1]) <= 0:
        msg = f"L-scale must be strictly positive, got lmda[1] = {l2}"
        raise InvalidLMomentError(msg)

    if len(l_r) <= 2:
        return

    # enforce the (non-strict) L-ratio bounds, from Hosking (2007) eq. 14,
    # but rewritten using falling factorials, to avoid potential overflows
    tau = l_r[2:] / l2

    r = np.arange(3, len(l_r) + 1)
    m = max(s, t) + 1
    tau_absmax = 2 * fpow(r + s + t, m) / (r * fpow(2 + s + t, m))

    if np.any(invalid := np.abs(tau) > tau_absmax):
        r_invalid = list(set(np.argwhere(invalid).ravel() + 3))
        if len(r_invalid) == 1:
            r_invalid = r_invalid[0]
        msg = f"L-moment(s) with r = {r_invalid} exceed the theoretical bounds"
        raise InvalidLMomentError(msg)

    # validate an l-skewness / l-kurtosis relative inequality that is
    # a pre-condition for the PPF to be strictly monotonically increasing
    t3 = tau[0]
    t4 = tau[1] if len(tau) > 1 else 0

    m = 2 + (s if t3 > 0 else t)
    u = 3 + s + t
    t3_max = 2 * (u / m + (m + 1) * (u + 4) * t4) / (3 * (u + 2))

    if abs(t3) >= t3_max:
        if t3 < 0:
            msg_t3_size, msg_t3_trim = "small", "the left trim order (S)"
        else:
            msg_t3_size, msg_t3_trim = "large", "the right trim order (t)"

        msg = (
            f"L-skewness is too {msg_t3_size} ({t3:.4f}); consider "
            f"increasing {msg_t3_trim}"
        )
        raise InvalidLMomentError(msg)
