__all__ = 'tl_weights', 'l_weights', 'reweight'

from math import comb, fsum
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from ._utils import expand_trimming
from .typing import Trimming


_T = TypeVar('_T', bound=npt.NBitBase)


def tl_weights(
    n: int,
    r: int,
    /,
    trim: Trimming = 1,
    *,
    dtype: type[np.floating[_T]] | np.dtype[np.floating[_T]] = np.float_,
) -> npt.NDArray[np.floating[_T]]:
    """
    Linear sample weights for calculation of the r-th TL-moment.

    Parameters:
        n: Sample size.
        r: L-moment order, e.g. 1 for location, and 2 for scale.
        trim:
            Amount of samples to trim as either

            - `(t1: int, t2: int)` for left and right trimming,
            - `t: int`, or `(t: int)` as alias for `(t, t)`, or
            - `()` as alias for `(0, 0)`.

            If not provided, `1` will be used by default.

    Other parameters:
        dtype:
            Desired output floating type for the weights, e.g,
            `numpy.float128`. Default is `numpy.float64`.
            Must be a (strict) subclass of `numpy.floating`.

    Returns:
        w_r:
            A vector of size `n`, with linear weights for each of the
            (ordered) samples.

    """

    if not issubclass(  # pyright: ignore [reportUnnecessaryIsInstance]
        np.dtype(dtype).type,
        np.floating
    ):
        raise TypeError(
            f'dtype must be a subclass of numpy.floating, got {dtype!r}'
        )

    if r <= 0:
        raise ValueError(f'expected r > 0, got {r} <= 0')

    tl, tr = expand_trimming(trim)

    if n < r + tl + tr:
        raise ValueError(f'expected n >= r + s + t, got {n} < {r + tl + tr}')

    # pre-calculate the terms that are independent on j
    m = r * comb(n, r + tl + tr)

    # https://github.com/numpy/numpy/issues/23783
    # w_k = np.empty(r, dtype=dtype)
    w_k = np.empty(r)

    for k in range(r):
        w_k[k] = (-1) ** k * comb(r - 1, k) / m

    # sample weights
    w_r = np.zeros(n, dtype=dtype)
    for j in range(tl, n - tr):
        # divide inside the loop, to avoid overflows
        w_r[j] = fsum(
            comb(j, r + tl - k - 1) * comb(n - j - 1, tr + k) * w_k[k]
            for k in range(r)
        )

    return w_r


def l_weights(
    n: int,
    r: int,
    /,
    *,
    dtype: type[np.floating[_T]] = np.float_,
) -> npt.NDArray[np.floating[_T]]:
    """
    Alias for [`tl_weights(n, r, 0)`][lmo.weights.tl_weights].
    """
    return tl_weights(n, r, 0, dtype=dtype)


def reweight(
    w_r: npt.NDArray[np.floating[_T]],
    w_x: npt.NDArray[np.bool_ | np.integer[Any] | np.floating[Any]],
) -> npt.NDArray[np.floating[_T]]:
    """
    Redistributes the TL-weights relative to the sample weights.

    Relatively large sample weights "absorb" TL-weights of the neighbours,
    whereas small weights result in a fraction of the local TL-weights.

    The TL-weights can be thought of as vertical bars, with heights
    proportional to the weights. Similarly, the sample weights are the
    horizontal component, effectively squeezing or stretching the width of each
    sample. The reweighted TL-weights are the resulting areas per sample.

    To my (Joren Hammudoglu, @jorenham) knowledge, this algorithm for weighted
    (T)L-moments is the first of its kind.

    Both time- and space- complexity are `O(n)`.

    Args:
        w_r:
            1-D array of TL-weights, see [`tl_weights`][lmo.weights.tl_weights].
        w_x:
            1-D array of observation (reliability) weights, relative to
            the *sorted* observations vector `x`. All weights must be finite
            and positive. Larger weights indicate a more important sample.
            If all weights are equal, the reweighted TL-weights will be equal
            to the original TL-weights.

    Returns:
        v_r: 1-D array of reweighted TL-weights.

    """
    if w_r.ndim != 1:
        raise TypeError('weights must be 1-D')
    if w_r.shape != w_x.shape:
        raise TypeError('shape mismatch')

    if np.all(w_r[0] == w_r):
        # all the same, e.g. for r=1 and trim=0
        s = w_x.sum() * w_r.sum()  # pyright: ignore [reportUnknownMemberType]
        return (w_x / s).astype(w_r.dtype)

    n = len(w_r)
    v_r = np.zeros_like(w_r)

    # integrate, and rescale so that `s.max() == s[-1] == n`
    s = np.cumsum(w_x, dtype=np.float_)
    s *= n / s[-1]

    n_j, s_j = 0, 0.0
    for k in range(n):
        s_k = s[k]

        if s_k < s_j:
            raise ValueError('negative weights are not allowed')
        if s_k == s_j:
            if s_k == s[-1]:
                break
            continue

        n_k = int(s_k)

        ds = s_k - s_j
        dn = n_k - n_j

        assert ds > 0
        assert 0 <= dn <= ds + 1

        if dn:
            ds_j = n_j + 1 - s_j
            ds_k = s_k - n_k

            assert 0 <= ds_j <= 1
            assert 0 <= ds_k < 1
            assert round(ds - ds_j - ds_k, 15) % 1 == 0

            # left partial indices
            v_r[k] = ds_j * w_r[n_j]

            # "inner" integer indices
            if dn > 1:
                assert ds > 1, (ds, dn)
                v_r[k] += np.sum(  # pyright: ignore [reportUnknownMemberType]
                    w_r[n_j + 1: n_k]
                )

            # right partial index
            if n_k < n:
                v_r[k] += ds_k * w_r[n_k]
        else:
            assert ds < 1
            assert n_j == n_k

            v_r[k] = ds * w_r[n_k]

        n_j, s_j = n_k, s_k

    return v_r
