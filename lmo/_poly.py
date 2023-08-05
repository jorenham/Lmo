"""
Helper functions for polynomials.

See Also:
    - [`numpy.polynomial`][numpy.polynomial]
    - [Classic orthogonal polynomials - Wikipedia
    ](https://wikipedia.org/wiki/Classical_orthogonal_polynomials)
"""

__all__ = (
    'jacobi',
    'jacobi_series',
    'roots',
    'extrema',
    'minima',
    'maxima',
)

from typing import Any, TypeVar

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt
import scipy.special as scs  # type: ignore

from .typing import PolySeries

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
) -> npp.Polynomial | npp.Legendre:
    r"""
    Construct a polynomial from the weighted sum of shifted Jacobi
    polynomials.

    Rougly equivalent to
    `sum(w[n] * sh_jacobi(n, a, b) for n in range(len(w)))`.

    Todo:
        - Create a `Jacobi` class, as extension to `numpy.polynomial.`
    """
    w = np.asarray(coef)
    if w.ndim != 1:
        msg = 'coefs must be 1-D'
        raise ValueError(msg)

    if a == b == 0:
        return npp.Legendre(w, domain=domain, window=window, symbol=symbol)

    n = len(w)
    return sum(
        w[r] * jacobi(r, a, b, domain=domain, window=window, symbol=symbol)
        for r in range(n) if abs(w[r]) > 1e-13
    ) # type: ignore


def roots(
    p: PolySeries,
    /,
    outside: bool = False,
) -> npt.NDArray[np.inexact[Any]]:
    """
    Return the $x$ in the domain of $p$, where $p(x) = 0$.

    If outside=False (default), the values that fall outside of the domain
    interval will be not be included.
    """
    x = p.roots()
    if not np.isrealobj(x) and np.isrealobj(p.domain):
        x = x[np.isreal(x)].real

    if not outside and len(x):
        a, b = np.sort(p.domain)
        return x[(x >= a) & (x <= b)]

    return x

def integrate(p: PolySeries, /, a: float | None = None) -> PolySeries:
    r"""Calculate the anti-derivative: $P(x) = \int_a^x p(u) \, du$."""
    return p.integ(lbnd=p.domain[0] if a is None else a)

def extrema(
    p: PolySeries,
    /,
    outside: bool = False,
) -> npt.NDArray[np.inexact[Any]]:
    """Return the $x$ in the domain of $p$, where $p'(x) = 0$."""
    return roots(p.deriv(), outside=outside)


def minima(
    p: PolySeries,
    /,
    outside: bool = False,
) -> npt.NDArray[np.inexact[Any]]:
    """
    Return the $x$ in the domain of $p$, where $p'(x) = 0$ and $p''(x) > 0$.
    """  # noqa: D200
    x = extrema(p, outside=outside)
    return x[p.deriv(2)(x) > 0] if len(x) else x


def maxima(
    p: PolySeries,
    /,
    outside: bool = False,
) -> npt.NDArray[np.inexact[Any]]:
    """
    Return the $x$ in the domain of $p$, where $p'(x) = 0$ and $p''(x) < 0$.
    """  # noqa: D200
    x = extrema(p, outside=outside)
    return x[p.deriv(2)(x) < 0] if len(x) else x
