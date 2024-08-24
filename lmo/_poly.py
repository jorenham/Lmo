"""
Helper functions for polynomials.

See Also:
    - [`numpy.polynomial`][numpy.polynomial]
    - [Classic orthogonal polynomials - Wikipedia
    ](https://wikipedia.org/wiki/Classical_orthogonal_polynomials)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast, overload

import numpy as np
import numpy.polynomial as npp
import optype.numpy as onpt
import scipy.special as scs
from numpy.polynomial._polybase import ABCPolyBase  # noqa: PLC2701


if TYPE_CHECKING:
    import lmo.typing.np as lnpt


__all__ = (
    'PolySeries',
    'eval_sh_jacobi',
    'peaks_jacobi',
    'arg_extrema_jacobi',
    'extrema_jacobi',
    'jacobi',
    'jacobi_series',
    'roots',
)


if TYPE_CHECKING:
    from typing_extensions import LiteralString

    PolySeries: TypeAlias = ABCPolyBase[LiteralString | None]
else:
    PolySeries: TypeAlias = ABCPolyBase


_T_shape = TypeVar('_T_shape', bound=onpt.AtLeast1D)
_T_poly = TypeVar('_T_poly', bound=PolySeries)


@overload
def eval_sh_jacobi(
    n: int,
    a: float | lnpt.Float,
    b: float | lnpt.Float,
    x: float | lnpt.Float,
) -> float: ...
@overload
def eval_sh_jacobi(
    n: int,
    a: float | lnpt.Float,
    b: float | lnpt.Float,
    x: onpt.Array[_T_shape, lnpt.Float],
) -> onpt.Array[_T_shape, np.float64]: ...
def eval_sh_jacobi(
    n: int | lnpt.Int,
    a: float | lnpt.Float,
    b: float | lnpt.Float,
    x: float | lnpt.Float | onpt.Array[_T_shape, lnpt.Float],
) -> float | onpt.Array[_T_shape, np.float64]:
    """
    Fast evaluation of the n-th shifted Jacobi polynomial.
    Faster than pre-computing using np.Polynomial, and than
    `scipy.special.eval_jacobi` for n < 4.
    """
    if n == 0:
        return 1

    x = np.asarray(x)[()]
    u = 2 * x - 1

    a = float(a)
    b = float(b)

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

        return scs.eval_sh_legendre(n, x)

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
    return scs.eval_jacobi(n, a, b, u)


def peaks_jacobi(
    n: int,
    a: float,
    b: float,
) -> onpt.Array[tuple[int], np.float64]:
    r"""
    Finds the \( x \in [-1, 1] \) s.t.
    \( /frac{\dd{\shjacobi{n}{a}{b}{x}}}{\dd{x}} = 0 \) of a Jacobi polynomial,
    which includes the endpoints \( x \in \{-1, 1\} \). I.e. the locations of
    the peaks.

    The Jacobi polynomials with order \( n \) have \( n + 1 \) peaks.

    Examples:
        For \( n = 0 \) there is only one "peak", since
        \( \jacobi{0}{a}{b}{x} = 1 \):

        >>> peaks_jacobi(0, 0, 0)
        array([0.])

        The `0` is arbitrary; all \( x \in [0, 1] \) evaluate to the same
        constant \( 1 \).

        For \( n = 1 \), it is a positive linear function, so the peaks
        are exactly the endpoints, and do not depend on \( a \) or \( b \):

        >>> peaks_jacobi(1, 0, 0)
        array([-1., 1.])
        >>> peaks_jacobi(1, 3.14, -1 / 12)
        array([-1., 1.])

        For \( n > 1 \), the effects of the choices for \( a \) and \( b \)
        become apparent, e.g. for \( n = 4 \):

        >>> peaks_jacobi(4, 0, 0).round(5)
        array([-1.     , -0.65465,  0.     ,  0.65465,  1.     ])
        >>> peaks_jacobi(4, 0, 1).round(5)
        array([-1.     , -0.50779,  0.1323 ,  0.70882,  1.     ])
        >>> peaks_jacobi(4, 1, 0).round(5)
        array([-1.     , -0.70882, -0.1323 ,  0.50779,  1.     ])
        >>> peaks_jacobi(4, 1, 1).round(5)
        array([-1.     , -0.57735,  0.     ,  0.57735,  1.     ])
        >>> peaks_jacobi(4, 2.5, 2.5)
        array([-1. , -0.5,  0. ,  0.5,  1. ])
        >>> peaks_jacobi(4, 10, 10).round(5)
        array([-1.     , -0.33333,  0.     ,  0.33333,  1.     ])
    """
    if n == 0:
        # constant; any x is a "peak"; so take the "middle ground"
        return np.array([0.])
    if n == 1:
        # linear; the peaks are only at the ends
        return np.array([-1., 1.])

    # otherwise, peaks are at the ends, and at the roots of the derivative
    x = np.empty(n + 1)
    x[0] = -1
    x[1:-1] = scs.roots_jacobi(n - 1, a + 1, b + 1)[0]  # pyright: ignore[reportUnknownMemberType]
    x[-1] = 1

    return np.round(x, 15) + 0.0  # cleanup of numerical noise


def arg_extrema_jacobi(n: int, a: float, b: float) -> tuple[float, float]:
    r"""
    Find the \( x \) of the minimum and maximum values of a Jacobi polynomial
    on \( [-1, 1] \).

    Note:
        There can be multiple \( x \) that share the same extremum, but only
        one of them is returned, which for \( n > 0 \) is the smallest (first)
        one.

    Examples:
        For \( n = 1 \), the Jacobi polynomials are positive linear function
        (i.e. a straight line), so the minimum and maximum are the left and
        right endpoints of the domain.

        >>> arg_extrema_jacobi(1, 0, 0)
        (-1.0, 1.0)
        >>> arg_extrema_jacobi(1, 3.14, -1 / 12)
        (-1.0, 1.0)

        The 2nd degree Jacobi polynomial is a positive parabola, with one
        unique minimum, and maxima at \( -1 \) and/or \( 1 \).
        When \( a == b \), the parabola is centered within the domain, and
        has maxima at both \( x = -1 \) and \( x=1 \). For the sake of
        simplicity, only one (the first) value is returned in such cases:

        >>> arg_extrema_jacobi(2, 0, 0)
        (0.0, -1.0)
        >>> arg_extrema_jacobi(2, 42, 42)
        (0.0, -1.0)

        Conversely, when \( a \neq b \), the parabola is "shifted" so that
        there is only one global maximum:

        >>> arg_extrema_jacobi(2, 0, 1)
        (0.2, -1.0)
        >>> arg_extrema_jacobi(2, 1, 0)
        (-0.2, 1.0)
        >>> arg_extrema_jacobi(2, 10, 2)
        (-0.5, 1.0)

    """
    x = peaks_jacobi(n, a, b)
    p = eval_sh_jacobi(n, a, b, (x + 1) / 2)

    return x[np.argmin(p)], x[np.argmax(p)]


def extrema_jacobi(n: int, a: float, b: float) -> tuple[float, float]:
    r"""
    Find the global minimum and maximum values of a (shifted) Jacobi
    polynomial on \( [-1, 1] \) (or equivalently \( [0, 1] \) if shifted).

    Examples:
        With \( n \), \( \jacobi{0}{a}{b}{x} = 1 \), so there is only one
        "extremum":

        >>> extrema_jacobi(0, 0, 0)
        (1, 1)
        >>> extrema_jacobi(0, 3.14, -1 / 12)
        (1, 1)

        With \( n = 1 \), the extrema are always at \( -1 \) and \( 1 \),
        but their values depend on \( a \) and \( b \):

        >>> extrema_jacobi(1, 0, 0)
        (-1.0, 1.0)
        >>> extrema_jacobi(1, 0, 1)
        (-2.0, 1.0)
        >>> extrema_jacobi(1, 1, 0)
        (-1.0, 2.0)
        >>> extrema_jacobi(1, 1, 1)
        (-2.0, 2.0)

        For \( n = 2 \) (a parabola), the relation between \( a, b \)
        and the extrema isn't as obvious:

        >>> extrema_jacobi(2, 0, 0)
        (-0.5, 1.0)
        >>> extrema_jacobi(2, 0, 4)
        (-0.75, 15.0)
        >>> extrema_jacobi(2, 4, 0)
        (-0.75, 15.0)
        >>> extrema_jacobi(2, 4, 4)
        (-1.5, 15.0)

        With \( n = 3 \), the extrema appear to behave very predictable:

        >>> extrema_jacobi(3, 0, 0)
        (-1.0, 1.0)
        >>> extrema_jacobi(3, 0, 1)
        (-4.0, 1.0)
        >>> extrema_jacobi(3, 1, 0)
        (-1.0, 4.0)
        >>> extrema_jacobi(3, 1, 1)
        (-4.0, 4.0)

        However, if we keep \( a \) fixed at \( 0 \), and increase \( b \),
        the plot-twist emerges:

        >>> extrema_jacobi(3, 0, 2)
        (-10.0, 1.0)
        >>> extrema_jacobi(3, 0, 3)
        (-20.0, 1.0)
        >>> extrema_jacobi(3, 0, 4)
        (-35.0, 1.13541)
        >>> extrema_jacobi(3, 0, 5)
        (-56.0, 1.25241)

        Looking at the corresponding \( x \) can help to understand the
        "movement" of the maximum.

        >>> arg_extrema_jacobi(3, 0, 2)
        (-1.0, 1.0)
        >>> arg_extrema_jacobi(3, 0, 3)
        (-1.0, 0.0)
        >>> arg_extrema_jacobi(3, 0, 4)
        (-1.0, 0.094495)
        >>> arg_extrema_jacobi(3, 0, 5)
        (-1.0, 0.172874)

    """
    x = peaks_jacobi(n, a, b)
    p = eval_sh_jacobi(n, a, b, (x + 1) / 2)
    return cast(float, np.min(p)), cast(float, np.max(p))


def _jacobi_coefs(
    n: int,
    a: float,
    b: float,
) -> onpt.Array[tuple[int], np.float64]:
    p_n: np.poly1d
    p_n = scs.jacobi(n, a, b)  # pyright: ignore[reportUnknownMemberType]
    return p_n.coef[::-1]


def jacobi(
    n: int,
    /,
    a: float,
    b: float,
    domain: tuple[float, float] = (-1, 1),
    window: tuple[float, float] = (-1, 1),
    symbol: str = 'x',
) -> npp.Polynomial:
    return npp.Polynomial(_jacobi_coefs(n, a, b), domain, window, symbol)


@overload
def jacobi_series(
    coef: lnpt.AnyArrayFloat,
    /,
    a: float,
    b: float,
    *,
    kind: None = ...,
    domain: tuple[float, float] = ...,
    window: tuple[float, float] = ...,
    symbol: str = ...,
) -> npp.Polynomial: ...
@overload
def jacobi_series(
    coef: lnpt.AnyArrayFloat,
    /,
    a: float,
    b: float,
    *,
    kind: type[_T_poly],
    domain: tuple[float, float] = ...,
    window: tuple[float, float] = ...,
    symbol: str = ...,
) -> _T_poly: ...
def jacobi_series(
    coef: lnpt.AnyArrayFloat,
    /,
    a: float,
    b: float,
    *,
    kind: type[_T_poly] | None = None,
    domain: tuple[float, float] = (-1, 1),
    window: tuple[float, float] = (-1, 1),
    symbol: str = 'x',
) -> _T_poly | npp.Polynomial:
    r"""
    Construct a polynomial from the weighted sum of shifted Jacobi
    polynomials.

    Roughly equivalent to
    `sum(w[n] * sh_jacobi(n, a, b) for n in range(len(w)))`.

    Todo:
        - Create a `Jacobi` class, as extension to `numpy.polynomial.`
    """
    w = cast(onpt.Array[tuple[int], np.float64], np.asarray(coef))
    if w.ndim != 1:
        msg = 'coefs must be 1-D'
        raise ValueError(msg)

    p = sum(
        jacobi(r, a, b, domain, window=window, symbol=symbol) * w_r
        for r, w_r in enumerate(w.flat)
    )
    assert p

    # see https://github.com/numpy/numpy/pull/27237
    return p.convert(  # pyright: ignore[reportUnknownMemberType]
        domain=domain,
        kind=kind or npp.Polynomial,
        window=window,
    )


def roots(
    p: PolySeries,
    /,
    outside: bool = False,
) -> onpt.Array[tuple[int], np.float64]:
    """
    Return the $x$ in the domain of $p$, where $p(x) = 0$.

    If outside=False (default), the values that fall outside of the domain
    interval will be not be included.
    """
    z = cast(
        onpt.Array[tuple[int], np.float64],
        p.roots(),
    )
    if not np.isrealobj(z) and np.isrealobj(p.domain):
        x = z[np.isreal(z)].real
    else:
        x = z

    if not outside and len(x):
        a, b = np.sort(p.domain)
        return x[(x >= a) & (x <= b)]

    return x
