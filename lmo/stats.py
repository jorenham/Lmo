__all__ = 'order_stats', 'l_ratio_max', 'pw_moment_cov', 'l_moment_cov'

from math import factorial as fact
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from ._utils import as_float_array, clean_order
from .linalg import sandwich, sh_legendre, hosking_jacobi
from .typing import IntVector, SortKind
from .weights import p_weights

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


def pw_moment_cov(
    a: npt.ArrayLike,
    ks: int,
    /,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    Distribution-free variance-covariance matrix of the $\\beta$ PWM point
    estimates with orders `k = 0, ..., ks`.

    Returns:
        S_b: Variance-covariance matrix/tensor of shape `(ks, ks, ...)`

    See Also:
        - https://wikipedia.org/wiki/Covariance_matrix

    References:
        - [E. Elmamir & A. Seheult (2004) - Exact variance structure of sample
            L-moments](https://doi.org/10.1016/S0378-3758(03)00213-1)

    """
    x = order_stats(a, axis=axis, dtype=dtype, **kwargs)

    # ensure the samples are "in front" (along axis=0)
    if axis and x.ndim > 1:
        x = np.moveaxis(x, axis, 0)

    n = len(x)
    P_k = p_weights(ks, n, dtype=dtype)

    # beta pwm estimates
    b = P_k @ x if x.ndim == 1 else np.inner(P_k, x.T)
    assert b.shape == (ks, *x.shape[1:])

    # the covariance matrix
    S_b = np.empty((ks, *b.shape), dtype=dtype)

    # dynamic programming approach for falling factorial ratios:
    # w_ki[k, i] = i^(k) / (n-1)^(k))
    ffact = np.ones((ks, n), dtype=dtype)
    for k in range(1, ks):
        ffact[k] = ffact[k - 1] * np.linspace((1 - k) / (n - k), 1, n)

    # ensure that at most ffact[..., -k_max] will give 0
    ffact = np.c_[ffact, np.zeros((ks, ks))]

    spec = 'i..., i...'

    # for k == l (variances on the diagonal):
    # sum(
    #     2 * i^(k) * (j-k-1)^(k) * x[i] * x[j]
    #     for j in range(i + 1, n)
    # ) / n^(k+l+2)
    for k in range(ks):
        j_k = np.arange(-k, n - k - 1)  # pyright: ignore
        v_ki = np.empty(x.shape, dtype=dtype)
        for i in range(n):
            # sum(
            #     2 * i^(k) * (j-k-1)^(k) * x[i] * x[j]
            #     for j in range(i + 1, n)
            # )
            v_ki[i] = ffact[k, j_k[i:]] * 2 * ffact[k, i] @ x[i + 1:]

        # (n-k-1)^(k+1)
        denom = n * (n - 2 * k - 1) * ffact[k, n - k - 1]
        m_bb = np.einsum(spec, v_ki, x) / denom  # pyright: ignore
        S_b[k, k] = b[k]**2 - m_bb

    # for k != l (actually k > l since symmetric)
    # sum(
    #     ( i^(k) * (j-k-1)^(l) + i^(l) * (j-l-1)^(k) )
    #     * x[i] * x[j]
    #     for j in range(i + 1, n)
    # ) / n^(k+l+2)
    for k, m in zip(*np.tril_indices(ks, -1)):
        j_k: npt.NDArray[np.int_] = np.arange(-k, n - k - 1)  # pyright: ignore
        j_l: npt.NDArray[np.int_] = np.arange(-m, n - m - 1)  # pyright: ignore

        v_ki = np.empty(x.shape, dtype=dtype)
        for i in range(n):
            v_ki[i] = (
                ffact[k, i] * ffact[m, j_k[i:]] +
                ffact[m, i] * ffact[k, j_l[i:]]
            ) @ x[i + 1:]

        # (n-k-1)^(l+1)
        denom = n * (n - k - m - 1) * ffact[m, n - k - 1]
        m_bb = np.einsum(spec, v_ki, x) / denom  # pyright: ignore

        # because s_bb.T == s_bb
        S_b[k, m] = S_b[m, k] = b[k] * b[m] - m_bb

    return S_b


def l_moment_cov(
    a: npt.ArrayLike,
    rs: int,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    Non-parmateric auto-covariance matrix of the generalized trimmed
    L-moment point estimates with orders `r = 1, ..., rs`.

    Returns:
        S_l: Variance-covariance matrix/tensor of shape `(rs, rs, ...)`

    References:
        - [E. Elmamir & A. Seheult (2004) - Exact variance structure of sample
            L-moments](https://doi.org/10.1016/S0378-3758(03)00213-1)

    """
    ks = rs + sum(trim)
    if ks < rs:
        raise ValueError('trimmings must be positive')

    # PWM covariance matrix
    S_b = pw_moment_cov(a, ks, axis=axis, dtype=dtype, **kwargs)

    # projection matrix: PWMs -> generalized trimmed L-moments
    P_l: npt.NDArray[np.floating[Any]]
    P_l = hosking_jacobi(rs, trim=trim, dtype=dtype) @ sh_legendre(ks)
    # clean some numerical noise
    P_l = np.round(P_l, 12) + 0.  # pyright: ignore [reportUnknownMemberType]

    # c'est ça
    return sandwich(P_l, S_b, dtype=dtype)
