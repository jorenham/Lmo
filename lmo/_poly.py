"""
Helper functions for polynomials.

See Also:
    - [`numpy.polynomial`][numpy.polynomial]
    - [Classic orthogonal polynomials - Wikipedia
    ](https://wikipedia.org/wiki/Classical_orthogonal_polynomials)
"""

__all__ = (
    'eval_sh_jacobi',
    'jacobi',
    'jacobi_series',
    'roots',
    'extrema',
    'minima',
    'maxima',
)

from typing import Any, TypeVar, cast, overload

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt
import scipy.special as scs  # type: ignore

from .typing import FloatVector, PolySeries

P = TypeVar('P', bound=PolySeries)


@overload
def eval_sh_jacobi(n: int, a: float, b: float, x: float) -> float:
    ...


@overload
def eval_sh_jacobi(
    n: int,
    a: float,
    b: float,
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    ...


def eval_sh_jacobi(
    n: int,
    a: float,
    b: float,
    x: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """
    Fast evaluation of the n-th shifted Jacobi polynomial.
    Faster than pre-computing using np.Polynomial, and than
    `scipy.special.eval_jacobi` for n < 4.
    """
    if n == 0:
        return 1

    u = 2 * x - 1

    if a == b == 0:
        if n == 1:
            return u

        v = x * (x - 1)

        if n == 2:
            return 1 + 6 * v
        if n == 3:
            return (1 + 10 * v) * u
        if n == 4:
            return 1 + 10 * v * (2 + 7 * v)

        return scs.eval_sh_legendre(n, x)  # type: ignore

    if n == 1:
        return (a + b + 2) * x - b - 1
    if n == 2:
        return (
            b * (b + 3)
            - (a + b + 3) * (
                2 * b + 4
                - (a + b + 4) * x
            ) * x
        ) / 2 + 1
    if n == 3:
        return (
            (1 + a) * (2 + a) * (3 + a)
            + (4 + a + b) * (
                3 * (2 + a) * (3 + a)
                + (5 + a + b) * (
                    3 * (3 + a)
                    + (6 + a + b) * (x - 1)
                ) * (x - 1)
            ) * (x - 1)
        ) / 6

    # don't use `eval_sh_jacobi`: https://github.com/scipy/scipy/issues/18988
    return scs.eval_jacobi(n, a, b, u)  # type: ignore


def _jacobi_coefs(n: int, a: float, b: float) -> npt.NDArray[np.float64]:
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

    Roughly equivalent to
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
) -> npt.NDArray[np.float64]:
    """
    Return the $x$ in the domain of $p$, where $p(x) = 0$.

    If outside=False (default), the values that fall outside of the domain
    interval will be not be included.
    """
    z = p.roots()
    if not np.isrealobj(z) and np.isrealobj(p.domain):
        x = z[np.isreal(z)].real
    else:
        x = cast(npt.NDArray[np.float64], z)

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
