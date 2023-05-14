__all__ = 'tl_weights', 'l_weights'

from math import comb, fsum

import numpy as np
import numpy.typing as npt

from ._utils import expand_trimming
from .typing import Trimming


def tl_weights(n: int, r: int, /, trim: Trimming) -> npt.NDArray[np.float_]:
    """
    Linear sample weights for calculation of the r-th TL(s, t)-moment.

    Args:
        n: Sample size.
        r: L-moment order, e.g. 1 for location, and 2 for scale.
        trim:
            Amount of samples to trim as either

            - `(t1: int, t2: int)` for left and right trimming,
            - `t: int`, or `(t: int)` as alias for `(t, t)`, or
            - `()` as alias for `(0, 0)`.

    Returns:
        w_j:
            A vector of size `n`, with linear weights for each of the
            (ordered) samples.

    """
    if r <= 0:
        raise ValueError(f'expected r > 0, got {r} <= 0')

    tl, tr = expand_trimming(trim)

    if n < r + tl + tr:
        raise ValueError(f'expected n >= r + s + t, got {n} < {r + tl + tr}')

    # pre-calculate the terms that are independent on j
    m = r * comb(n, r + tl + tr)
    w_k = np.empty(r)
    for k in range(r):
        w_k[k] = (-1) ** k * comb(r - 1, k) / m

    # sample weights
    w_j = np.zeros(n)
    for j in range(tl, n - tr):
        # divide inside the loop, to avoid overflows
        w_j[j] = fsum(
            comb(j, r + tl - k - 1) * comb(n - j - 1, tr + k) * w_k[k]
            for k in range(r)
        )

    return w_j


def l_weights(n: int, r: int, /) -> npt.NDArray[np.float_]:
    """
    Alias for `tl_weights(n, r, 0, 0)`.
    """
    return tl_weights(n, r, 0)
