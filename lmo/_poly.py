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

from typing import Any, TypeVar, cast

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt
import scipy.special as scs  # type: ignore

from .typing import FloatVector, PolySeries

P = TypeVar('P', bound=PolySeries)


def _jacobi_coefs(n: int, a: float, b: float) -> npt.NDArray[np.float_]:
    p_n: np.poly1d
    p_n = scs.jacobi(n, a, b)  # type: ignore [reportUnknownMemberType]
    return p_n.coef[::-1]


def jacobi(
    n: int,
    /,
    a: float,
    b: float,
    domain: FloatVector = (-1, 1),
    window: FloatVector = (-1, 1),
    symbol: str = 'x',
) -> npp.Polynomial:
    return npp.Polynomial(_jacobi_coefs(n, a, b), domain, window, symbol)


def jacobi_series(
    coef: npt.ArrayLike,
    /,
    a: float,
    b: float,
    *,
    domain: FloatVector = (-1, 1),
    kind: type[P] | None = None,
    window: FloatVector = (-1, 1),
    symbol: str = 'x',
) -> P:
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

    # if a == b == 0:
    #     p = npp.Legendre(w, symbol=symbol, **kwargs)
    # else:
    n = len(w)
    p = cast(
        PolySeries,
        sum(
            w[r] * jacobi(r, a, b, domain=domain, symbol=symbol, window=window)
            for r in range(n)
        ),
    )

    return cast(P, p.convert(domain=domain, kind=kind, window=window))


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
