__all__ = 'l_weights',

import numpy as np
import numpy.typing as npt

from ._meta import jit
from .special import sh_legendre


@jit
def l_weights(n: int, k_max: int, /) -> npt.NDArray[np.float_]:
    """
    L-moment linear sample weights.

    Equivalent to (but numerically more unstable than) this "naive" version::

        w = np.zeros((n, k))
        for r in range(1, n+1):
            for l in range(1, k+1):
                for j in range(min(r, l)):
                    w[r-1, l-1] += (
                        (-1) ** (l-1-j)
                        * comb(l-1, j) * comb(l-1+j, j)
                        * comb(r-1, j) / comb(n-1, j)
                    )
        return w / n

    Args:
        n: Amount of observations.
        k_max: Max degree of the L-moment, s.t. `0 <= k < n`.

    Returns:
        w: Weights of shape (n, k)

    """
    assert 0 < k_max < n

    w_kn = np.zeros((n, k_max), np.float_)
    w_kn[:, 0] = 1 / n
    for j in range(k_max - 1):
        w_kn[j:, j+1] = w_kn[j:, j] * np.arange(0, n-j, 1, np.float_) / (n-j)

    return w_kn @ sh_legendre(k_max).astype(np.float_).T
