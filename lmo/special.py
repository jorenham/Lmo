"""Mathematical "special" functions, extending `scipy.special`."""
from __future__ import annotations

from typing import Any, Final, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy.special as sc

import lmo.typing as lmt
import lmo.typing.np as lnpt
from ._utils import clean_orders


__all__ = 'fpow', 'gamma2', 'harmonic', 'norm_sh_jacobi', 'fourier_jacobi'


_DTYPE_CHARS: Final[str] = '?bBhHiIlLqQpP'

_T_shape = TypeVar('_T_shape', bound=onpt.AtLeast1D)
_T_float = TypeVar('_T_float', bound=lnpt.Float)


@overload
def fpow(
    x: lnpt.AnyScalarFloat,
    n: lnpt.AnyScalarFloat,
    /,
    out: None = ...,
) -> np.float64: ...
@overload
def fpow(
    x: lnpt.AnyArrayFloat,
    n: lnpt.AnyScalarFloat | lnpt.AnyArrayFloat,
    /,
    out: None = ...,
) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
@overload
def fpow(
    x: lnpt.AnyScalarFloat,
    n: lnpt.AnyArrayFloat,
    /,
    out: None = ...,
) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
@overload
def fpow(
    x: lnpt.AnyArrayFloat,
    n: lnpt.AnyScalarFloat | lnpt.AnyArrayFloat,
    /,
    out: onpt.Array[_T_shape, _T_float],
) -> onpt.Array[_T_shape, _T_float]: ...
@overload
def fpow(
    x: lnpt.AnyScalarFloat,
    n: lnpt.AnyArrayFloat,
    /,
    out: onpt.Array[_T_shape, _T_float],
) -> onpt.Array[_T_shape, _T_float]: ...
def fpow(
    x: lnpt.AnyScalarFloat | lnpt.AnyArrayFloat,
    n: lnpt.AnyScalarFloat | lnpt.AnyArrayFloat,
    /,
    out: onpt.Array[_T_shape, _T_float] | None = None,
) -> (
    np.float64
    | onpt.Array[_T_shape, _T_float]
    | onpt.Array[onpt.AtLeast1D, np.float64]
):
    r"""
    Factorial power, or falling factorial.

    It is defined as

    \[
        \ffact{x}{n} = \frac{\Gamma(x + 1)}{\Gamma(x - n + 1)}
    \]

    Args:
        x: Real-valued array-like or scalar.
        n: Real valued array-like or scalar.
        out: Optional output array for the function results

    Returns:
        out: Array or scalar with the value(s) of the function.

    See Also:
        - [`scipy.special.poch`][scipy.special.poch] -- the rising factorial
    """
    _x, _n = np.asanyarray(x), np.asanyarray(n)
    res = cast(npt.NDArray[np.float64], sc.poch(_x - _n + 1, _n, out=out))
    if res.ndim == 0 and np.isscalar(x) and np.isscalar(n):
        return res[()]
    return res


@overload
def gamma2(
    a: lnpt.AnyScalarFloat,
    x: lnpt.AnyScalarFloat,
    /,
    out: None = ...,
) -> np.float64: ...
@overload
def gamma2(
    a: lnpt.AnyScalarFloat,
    x: lnpt.AnyArrayFloat,
    /,
    out: None = ...,
) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
@overload
def gamma2(
    a: lnpt.AnyScalarFloat,
    x: lnpt.AnyArrayFloat,
    /,
    out: onpt.Array[_T_shape, _T_float],
) -> onpt.Array[_T_shape, _T_float]: ...
def gamma2(
    a: lnpt.AnyScalarFloat,
    x: lnpt.AnyScalarFloat | lnpt.AnyArrayFloat,
    /,
    out: onpt.Array[_T_shape, _T_float] | None = None,
) -> (
    np.float64
    | onpt.Array[_T_shape, _T_float]
    | onpt.Array[onpt.AtLeast1D, np.float64]
):
    r"""
    Incomplete (upper) gamma function.

    It is defined as

    \[
        \Gamma(a,\ x) = \int_x^\infty t^{a-1} e^{-t} \mathrm{d}t
    \]

    for \( a \ge 0 \) and \( x \ge 0 \).

    Args:
        a: Real-valued non-negative scalar.
        x: Real-valued non-negative array-like.
        out: Optional output array for the results.

    Returns:
        out: Scalar of array with the values of the incomplete gamma function.

    See Also:
        - [`scipy.special.gammaincc`][scipy.special.gammaincc] for the
          regularized gamma function \( Q(a,\ x) \).
    """
    if a == 0:
        return sc.exp1(x, out=out)
    return sc.gammaincc(a, x, out=out) * sc.gamma(a)


@overload
def harmonic(n: lnpt.AnyScalarFloat, /, out: None = ...) -> float: ...
@overload
def harmonic(
    n: lnpt.AnyArrayFloat,
    /,
    out: None = ...,
) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
@overload
def harmonic(
    n: lnpt.AnyArrayFloat,
    /,
    out: onpt.Array[_T_shape, _T_float],
) -> onpt.Array[_T_shape, _T_float]: ...
def harmonic(
    n: lnpt.AnyScalarFloat | lnpt.AnyArrayFloat,
    /,
    out: onpt.Array[_T_shape, _T_float] | None = None,
) -> (
    float
    | onpt.Array[_T_shape, _T_float]
    | onpt.Array[onpt.AtLeast1D, np.float64]
):
    r"""
    Harmonic number \( H_n = \sum_{k=1}^{n} 1 / k \), extended for real and
    complex argument via analytic contunuation.

    Examples:
        >>> harmonic(0)
        0.0
        >>> harmonic(1)
        1.0
        >>> harmonic(2)
        1.5
        >>> harmonic(42)
        4.32674
        >>> harmonic(np.pi)
        1.87274
        >>> harmonic(-1 / 12)
        -0.146106

    Args:
        n: Real- or complex- valued parameter, as array-like or scalar.
        out: Optional real or complex output array for the results.

    Returns:
        out: Array or scalar with the value(s) of the function.

    See Also:
        - [Harmonic number - Wikipedia](https://w.wiki/A63b)
    """
    _n = np.asanyarray(n)

    _out = sc.digamma(_n + 1, out) + np.euler_gamma
    return float(_out[()]) if np.isscalar(n) else _out


@overload
def norm_sh_jacobi(
    n: lmt.AnyOrder,
    alpha: float,
    beta: float,
) -> np.float64: ...
@overload
def norm_sh_jacobi(
    n: lmt.AnyOrderND,
    alpha: float,
    beta: float,
) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
def norm_sh_jacobi(
    n: lmt.AnyOrder | lmt.AnyOrderND,
    alpha: float,
    beta: float,
) -> np.float64 | onpt.Array[onpt.AtLeast1D, np.float64]:
    r"""
    Evaluate the (weighted) \( L^2 \)-norm of a shifted Jacobi polynomial.

    Specifically,

    \[
        \| p_n \|^2
        = \braket{p_n | p_n}
        = \int_0^1 |p_n|^2 \mathrm{d}x
        = \frac{1}{2 n + \alpha + \beta + 1} \frac
            {\Gamma(n + \alpha + 1) \Gamma(n + \beta + 1)}
            {n! \ \Gamma(n + \alpha + \beta + 1)}
    \]

    with

    \[
        p_n(x) \equiv
            x^{\beta / 2} \
            (1 - x)^{\alpha / 2} \
            \shjacobi{n}{\alpha}{\beta}{x}
    \]

    the normalized Jacobi polynomial on \( [0, 1] \).
    """
    if alpha <= -1:
        msg = f'alpha must be > -1, got {alpha}'
        raise ValueError(msg)
    if beta <= -1:
        msg = f'beta must be > -1, got {beta}'
        raise ValueError(msg)

    r = clean_orders(np.asanyarray(n), 'n') + 1

    if alpha == beta == 0:
        # shifted Legendre
        c = np.ones(r.shape)
    elif alpha == beta == -1 / 2:
        # shifted Chebychev of the first kind
        c = np.exp(2 * (sc.gammaln(r - 1 / 2) - sc.gammaln(r))) / 2
    elif alpha == beta == 1 / 2:
        # shifted Chebychev of the second kind
        c = np.exp(2 * (sc.gammaln(r + 1 / 2) - sc.gammaln(r + 1))) / 2
    else:
        p, q = r + alpha, r + beta
        c = np.exp(sc.betaln(p, q) - sc.betaln(r, p + beta)) / (p + q - 1)

    return c[()] if r.ndim == 0 and np.isscalar(n) else c


@overload
def fourier_jacobi(
    x: lnpt.AnyArrayFloat,
    c: lnpt.AnyArrayFloat,
    a: float,
    b: float,
) -> onpt.Array[Any, np.float64]: ...
@overload
def fourier_jacobi(
    x: lnpt.AnyScalarFloat,
    c: lnpt.AnyArrayFloat,
    a: float,
    b: float,
) -> np.float64: ...
def fourier_jacobi(
    x: lnpt.AnyScalarFloat | lnpt.AnyArrayFloat,
    c: lnpt.AnyArrayFloat,
    a: float,
    b: float,
) -> np.float64 | onpt.Array[onpt.AtLeast1D, np.float64]:
    r"""
    Evaluate the Fourier-Jacobi series, using the Clenshaw summation
    algorithm.

    If \( c \) is of length \( n + 1 \), this function returns the value:

    \[
        c_0 \cdot \jacobi{0}{a}{b}{x} +
        c_1 \cdot \jacobi{1}{a}{b}{x} +
        \ldots +
        c_n \cdot \jacobi{n}{a}{b}{x}
    \]

    Here, \( \jacobi{n}{a}{b}{x} \) is a Jacobi polynomial of degree
    \( n = |\vec{c}| \), which is orthogonal iff
    \( (a, b) \in (-1,\ \infty)^2 \) and \( x \in [0,\ 1] \).

    Tip:
        Trailing zeros in the coefficients will be used in the evaluation,
        so they should be avoided if efficiency is a concern.

    Args:
        x: Scalar or array-like with input data.
        c:
            Array-like of coefficients, ordered from low to high. All
            coefficients to the right are considered zero.

            For instance, `[4, 3, 2]` gives \( 4 \jacobi{0}{a}{b}{x} +
            3 \jacobi{1}{a}{b}{x} + 2 \jacobi{2}{a}{b}{x} \).
        a: Jacobi parameter \( a > -1 \).
        b: Jacobi parameter \( a > -1 \).

    Returns:
        out: Scalar or array of same shape as `x`.

    See Also:
        - [Generalized Fourier series - Wikipedia](
        https://wikipedia.org/wiki/Generalized_Fourier_series)
        - [Clenshaw Recurrence Formula - Wolfram MathWorld](
        https://mathworld.wolfram.com/ClenshawRecurrenceFormula.html)
        - [Jacobi Polynomial - Worlfram Mathworld](
        https://mathworld.wolfram.com/JacobiPolynomial.html)
    """
    _c = cast(
        npt.NDArray[np.integer[Any] | np.floating[Any]],
        np.array(c, ndmin=1, copy=2),  # pyright: ignore[reportCallIssue,reportArgumentType]
    )
    if _c.dtype.char in _DTYPE_CHARS:
        _c = _c.astype(np.float64)

    _x = np.asanyarray(x)

    if len(_c) == 0:
        return 0. * _x

    # temporarily replace inf's with abs(_) > 1, and track the sign
    if hasinfs := np.any(infs := np.isinf(_x)):
        _x = np.where(infs, 10 * np.sign(_x), _x)

    # "backwards" recursion (left-reduction)
    # y[k+2] and y[k+1]
    y2, y1 = 0., 0.
    # continue until y[0]
    for k in range(len(_c) - 1, 0, -1):
        # Jacobi recurrence terms
        u, v = a + k, b + k
        w = u + v  # = a + b + 2*k
        # alpha[k]
        p1 = (
            (w + 1) * (
                2 * k * (v - u)
                + w * (u - v + (w + 2) * _x)
            )
            / (2 * w * (k + 1) * (w - k + 1))
        )
        # beta[k+1]
        q2 = -(
            (u + 1) * (v + 1) * (w + 4)
            / ((w + 2) * (k + 2) * (w - k + 2))
        )

        # update the state; "forget" y[k+2]
        y1, y2 = _c[k] + p1 * y1 + q2 * y2, y1

    # results of jacobi polynomial for n=0 and n=1
    f0 = 1
    f1 = (a - b + (a + b + 2) * _x) / 2

    # beta[1]
    q1 = -(
        (a + 1) * (b + 1) * (a + b + 4)
        / (2 * (a + b + 2)**2)
    )

    # Behold! The magic of Clenshaw's algorithm:
    out: npt.NDArray[np.float64] = _c[0] * f0 + y1 * f1 + y2 * q1 * f0

    # propagation of 'inf' values, ensuring correct sign
    if hasinfs:
        out[infs] = np.inf * np.sign(out[infs] - _c[0] * f0)

    # unpack array iff `x` is scalar; 0-d arrays will pass through
    return out[()] if np.isscalar(x) else out
