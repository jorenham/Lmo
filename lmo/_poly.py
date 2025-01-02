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
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc
import scipy.special as sps
from numpy.polynomial._polybase import ABCPolyBase  # noqa: PLC2701

if TYPE_CHECKING:
    from typing import LiteralString


__all__ = (
    "PolySeries",
    "arg_extrema_jacobi",
    "eval_sh_jacobi",
    "extrema_jacobi",
    "jacobi",
    "jacobi_series",
    "peaks_jacobi",
    "roots",
)


def __dir__() -> tuple[str, ...]:
    return __all__


if TYPE_CHECKING:
    PolySeries: TypeAlias = ABCPolyBase[LiteralString | None]
else:
    PolySeries: TypeAlias = ABCPolyBase


_T_poly = TypeVar("_T_poly", bound=PolySeries)
_Float = np.float32 | np.float64

###


@overload
def eval_sh_jacobi(
    n: int | npc.integer,
    a: float,
    b: float,
    x: onp.ToFloat,
) -> float: ...
@overload
def eval_sh_jacobi(
    n: int | npc.integer,
    a: float,
    b: float,
    x: onp.ArrayND[_Float],
) -> onp.ArrayND[_Float]: ...
def eval_sh_jacobi(  # noqa: C901
    n: int | npc.integer,
    a: float,
    b: float,
    x: onp.ToFloat | onp.ArrayND[_Float],
) -> float | onp.ArrayND[_Float]:
    """
    Fast evaluation of the n-th shifted Jacobi polynomial.
    Faster than pre-computing using np.Polynomial, and than
    `scipy.special.eval_jacobi` for n < 4.
    """
    if n == 0:
        return 1

    x = x if isinstance(x, np.ndarray) else float(x)

    if a == b == 0:
        if n == 1:
            return 2 * x - 1
        if n > 4:
            out = sps.eval_sh_legendre(n, x)
            return float(out) if isinstance(out, np.floating) else out

        v = x * (x - 1)

        if n == 2:
            return 1 + 6 * v
        if n == 3:
            return (1 + 10 * v) * (2 * x - 1)
        if n == 4:
            return 1 + 10 * v * (2 + 7 * v)

    if n == 1:
        return (a + b + 2) * x - b - 1
    if n == 2:
        return (b * (b + 3) - (a + b + 3) * (2 * b + 4 - (a + b + 4) * x) * x) / 2 + 1
    if n == 3:
        return (
            (1 + a) * (2 + a) * (3 + a)
            + (4 + a + b)
            * (
                3 * (2 + a) * (3 + a)
                + (5 + a + b) * (3 * (3 + a) + (6 + a + b) * (x - 1)) * (x - 1)
            )
            * (x - 1)
        ) / 6

    # don't use `eval_sh_jacobi`: https://github.com/scipy/scipy/issues/18988
    return sps.eval_jacobi(n, a, b, 2 * x - 1)


def peaks_jacobi(n: int, a: float, b: float) -> onp.ArrayND[_Float]:
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
        return np.array([0.0], np.float64)
    if n == 1:
        # linear; the peaks are only at the ends
        return np.array([-1.0, 1.0])

    # otherwise, peaks are at the ends, and at the roots of the derivative
    x = np.empty(n + 1)
    x[0] = -1
    x[1:-1] = sps.roots_jacobi(n - 1, a + 1, b + 1)[0]
    x[-1] = 1

    return np.round(x, 15) + 0.0


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
    return cast("float", np.min(p)), cast("float", np.max(p))


def jacobi(
    n: onp.ToInt,
    /,
    a: onp.ToFloat,
    b: onp.ToFloat,
    domain: tuple[onp.ToFloat, onp.ToFloat] = (-1, 1),
    window: tuple[onp.ToFloat, onp.ToFloat] = (-1, 1),
    symbol: str = "x",
) -> npp.Polynomial:
    return npp.Polynomial(
        sps.jacobi(n, a, b).coef[::-1],
        domain=domain,
        window=window,
        symbol=symbol,
    )


@overload
def jacobi_series(
    coef: onp.ToFloat1D,
    /,
    a: onp.ToFloat,
    b: onp.ToFloat,
    *,
    kind: None = ...,
    domain: tuple[onp.ToFloat, onp.ToFloat] = ...,
    window: tuple[onp.ToFloat, onp.ToFloat] = ...,
    symbol: str = ...,
) -> npp.Polynomial: ...
@overload
def jacobi_series(
    coef: onp.ToFloat1D,
    /,
    a: onp.ToFloat,
    b: onp.ToFloat,
    *,
    kind: type[_T_poly],
    domain: tuple[onp.ToFloat, onp.ToFloat] = ...,
    window: tuple[onp.ToFloat, onp.ToFloat] = ...,
    symbol: str = ...,
) -> _T_poly: ...
def jacobi_series(
    coef: onp.ToFloat1D,
    /,
    a: onp.ToFloat,
    b: onp.ToFloat,
    *,
    kind: type[_T_poly] | None = None,
    domain: tuple[onp.ToFloat, onp.ToFloat] = (-1, 1),
    window: tuple[onp.ToFloat, onp.ToFloat] = (-1, 1),
    symbol: str = "x",
) -> _T_poly | npp.Polynomial:
    """
    Construct a polynomial from the weighted sum of shifted Jacobi
    polynomials.

    Roughly equivalent to `sum(wn * sh_jacobi(r, a, b) for r, wn in enumerate(w))`.

    Todo:
        - Create a `Jacobi` class, as extension to `numpy.polynomial.`
    """
    w = np.asarray(coef, np.float64)
    if w.ndim != 1:
        msg = "coefs must be 1-D"
        raise ValueError(msg)

    p = sum(
        wn * jacobi(n, a, b, domain=domain, window=window, symbol=symbol)
        for n, wn in enumerate(w.flat)
    )
    assert p

    return p.convert(domain=domain, kind=kind or npp.Polynomial, window=window)


def roots(p: PolySeries, /, outside: op.CanBool = False) -> onp.ArrayND[np.float64]:
    """
    Return the $x$ in the domain of $p$, where $p(x) = 0$.

    If outside=False (default), the values that fall outside of the domain
    interval will be not be included.
    """
    z = p.roots()
    x = cast(
        "onp.Array1D[np.float64]",
        z[np.isreal(z)].real if np.isrealobj(p.domain) and not np.isrealobj(z) else z,
    )

    if not outside and x.size:
        a, b = np.sort(p.domain)
        return x[(x >= a) & (x <= b)]

    return x
