__all__ = 'l_weights',

import numpy as np
import numpy.typing as npt

from ._meta import jit
from .special import sh_legendre


@jit
def l_weights(k_max: int, n: int, /) -> npt.NDArray[np.float_]:
    """
    L-moment linear sample weights.

    Equivalent to (but numerically more unstable than) this "naive" version::

        w = np.zeros((k, n))
        for r in range(n):
            for l in range(k):
                for j in range(min(r, l) + 1):
                    w[l, r] += (
                        (-1) ** (l - j)
                        * comb(l, j) * comb(l + j, j)
                        * comb(r, j) / comb(n - 1, j)
                    )
        return w / n

    Args:
        k_max: Max degree of the L-moment, s.t. `0 <= k < n`.
        n: Amount of observations.

    Returns:
        w: Weights of shape (k_max, n)

    """
    assert 0 < k_max <= n

    w_kn = np.zeros((k_max, n))
    w_kn[0] = 1 / n
    for j in range(k_max - 1):
        w_kn[j + 1, j:] = w_kn[j, j:] * np.linspace(0, 1 - 1 / (n - j), n - j)

    return sh_legendre(k_max).astype(np.float_) @ w_kn
