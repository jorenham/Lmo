__all__ = 'tl_weights', 'l_weights'

from math import comb, fsum
from typing import TypeVar

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
    dtype: type[np.floating[_T]] = np.float_,
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

    if not issubclass(
        dtype,
        np.floating
    ):  # pyright: ignore [reportUnnecessaryIsInstance]
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
    w_k = np.empty(r, dtype=dtype)
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
