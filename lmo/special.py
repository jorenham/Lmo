__all__ = (
    'sh_legendre',
)

from typing import Any

import numpy as np
import numpy.typing as npt

from ._meta import jit


@jit
def sh_legendre(m: int | np.integer[Any], /) -> npt.NDArray[np.int_]:
    """
    Shifted legendre polynomial coefficient matrix with shape (m, m), where
    the ``k``-th coefficient of the shifted Legendre polynomial of degree ``n``
    is at ``(n, k)``::

        P[n, k]
            = (-1)**(n-k) * comb(n, k) * comb(n + k, k)
            = scipy.special.sh_legendre(i).coef[i - j]


    Useful for broadcasting shifted legendre coefficients, e.g. for the
    calculation of probability-weighted moments.

    Implemented as the elementwise product of of the symmetric Pascal matrix
    with the inverse of the lower Pascal matrix of size (m + 1, m).

    Args:
        m: Order of the shifted legendre polynomial, >0.

    Returns:
        P: Integer array of shape `(m, m)` with shifted legendre coefficients.

    See Also:
        * https://wikipedia.org/wiki/Legendre_polynomials
        * https://wikipedia.org/wiki/Pascal_matrix

    """
    if m < 0:
        raise ValueError

    # Simultaneously calculate the lower- and symmetric inverse Pascal matrices.
    lp = np.zeros((m, m), np.int_)
    p2 = np.ones((m, m), np.int_)

    lp[0, 0] = 1
    for i in range(1, m):
        lp[i, 0] = (-1) ** i
        for j in range(1, m):
            lp[i, j] = lp[i - 1, j - 1] - lp[i - 1, j]
            p2[i, j] = p2[i - 1, j] + p2[i, j - 1]

    # Behold! Mathemagic...
    return lp * p2
