"""
Helper functions for classic orthogonal polynomials.

See Also:
    - [Classic orthogonal polynomials - Wikipedia
    ](https://wikipedia.org/wiki/Classical_orthogonal_polynomials)
"""

__all__ = ('jacobi', 'jacobi_series')

from typing import Any, TypeVar

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt
import scipy.special as scs  # type: ignore

T = TypeVar('T', bound=np.floating[Any] | np.object_)


def _jacobi_coefs(n: int, a: float, b: float) -> npt.NDArray[np.float_]:
    p_n: np.poly1d
    p_n = scs.jacobi(n, a, b)  # type: ignore [reportUnknownMemberType]
    return p_n.coef[::-1]


def jacobi(
    n: int,
    /,
    a: float,
    b: float,
    domain: npt.ArrayLike = (-1, 1),
    window: npt.ArrayLike = (-1, 1),
    symbol: str = 'x',
) -> npp.Polynomial:
    return npp.Polynomial(_jacobi_coefs(n, a, b), domain, window, symbol)


def jacobi_series(
    coef: npt.ArrayLike,
    /,
    a: float,
    b: float,
    *,
    domain: npt.ArrayLike = (-1, 1),
    window: npt.ArrayLike = (-1, 1),
    symbol: str = 'x',
) -> npp.Polynomial:
    r"""
    Construct a polynomial from the weighted sum of shifted Jacobi
    polynomials.

    Rougly equivalent to
    `sum(w[n] * sh_jacobi(n, a, b) for n in range(len(w)))`.
    """
    w = np.asarray(coef)
    if w.ndim != 1:
        msg = 'coefs must be 1-D'
        raise ValueError(msg)

    n = len(w)
    return sum(
        w[r] * jacobi(r, a, b, domain=domain, window=window, symbol=symbol)
        for r in range(n) if abs(w[r]) > 1e-13
    ) # type: ignore
