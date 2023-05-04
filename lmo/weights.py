__all__ = 'tl_weights', 'l_weights'

from math import comb, fsum

import numpy as np
import numpy.typing as npt


def tl_weights(n: int, r: int, /, s: int, t: int) -> npt.NDArray[np.float_]:
    """
    Linear sample weights for calculation of the r-th TL(s, t)-moment.

    Args:
        n: Sample size.
        r: L-moment order, e.g. 1 for location, and 2 for scale.
        s: Amount of samples to trim at the start/left.
        t: Amount of samples to trim at the end/right.

    Returns:
        w_j: A vector of size `n`, with linear weights for each of the
            (ordered) samples.

    """
    if r <= 0:
        raise ValueError(f'expected r > 0, got {r} <= 0')
    if min(s, t) < 0:
        raise ValueError(f'expected s >= 0 and t >= 0, got min{s, t} < 0')
    if n < r + s + t:
        raise ValueError(f'expected n >= r + s + t, got {n} < {r + s + t}')

    # pre-calculate the terms that are independent on j
    m = r * comb(n, r + s + t)
    w_k = np.empty(r)
    for k in range(r):
        w_k[k] = (-1) ** k * comb(r - 1, k) / m

    # sample weights
    w_j = np.zeros(n)
    for j in range(s, n - t):
        # divide inside the loop, to avoid overflows
        w_j[j] = fsum(
            comb(j, r + s - k - 1) * comb(n - j - 1, t + k) * w_k[k]
            for k in range(r)
        )

    return w_j


def l_weights(n: int, r: int, /) -> npt.NDArray[np.float_]:
    """
    Alias for `tl_weights(n, r, 0, 0)`.
    """
    return tl_weights(n, r, 0, 0)
