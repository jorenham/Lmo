"""Mathematical "special" functions, extending `scipy.special`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
import scipy.special as sps

from ._utils import clean_orders

if TYPE_CHECKING:
    import lmo.typing as lmt

__all__ = "fourier_jacobi", "fpow", "gamma2", "harmonic", "norm_sh_jacobi"


def __dir__() -> tuple[str, ...]:
    return __all__


_OutT = TypeVar("_OutT", bound=onp.ArrayND[np.number[Any]])

_Float: TypeAlias = float | np.float32 | np.float64
_FloatND: TypeAlias = onp.ArrayND[np.float32 | np.float64]


###

_DTYPE_CHARS: Final = "?bBhHiIlLqQpP"


@overload
def fpow(x: onp.ToFloat, n: onp.ToFloat, /, out: None = None) -> float: ...
@overload
def fpow(x: onp.ToFloat, n: onp.ToFloatND, /, out: None = None) -> _FloatND: ...
@overload
def fpow(
    x: onp.ToFloatND,
    n: onp.ToFloat | onp.ToFloatND,
    /,
    out: None = None,
) -> _FloatND: ...
@overload
def fpow(
    x: onp.ToFloat | onp.ToFloatND,
    n: onp.ToFloat | onp.ToFloatND,
    /,
    out: _OutT,
) -> _OutT: ...
def fpow(
    x: onp.ToFloat | onp.ToFloatND,
    n: onp.ToFloat | onp.ToFloatND,
    /,
    out: _OutT | None = None,
) -> float | _FloatND | _OutT:
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
    x_: _FloatND = np.asanyarray(x)
    n_: _FloatND = np.asanyarray(n)

    if out is not None:
        return sps.poch(x_ - n_ + 1, n_, out=out)

    res = sps.poch(x_ - n_ + 1, n_)
    return res.item() if res.ndim == 0 and np.isscalar(x) and np.isscalar(n) else res


@overload
def gamma2(a: onp.ToFloat, x: onp.ToFloat, /, out: None = None) -> _Float: ...
@overload
def gamma2(a: onp.ToFloat, x: onp.ToFloatND, /, out: None = None) -> _FloatND: ...
@overload
def gamma2(a: onp.ToFloat, x: onp.ToFloatND, /, out: _OutT) -> _OutT: ...
def gamma2(
    a: onp.ToFloat,
    x: onp.ToFloat | onp.ToFloatND,
    /,
    out: _OutT | None = None,
) -> _OutT | _Float | _FloatND:
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
        return sps.expm1(x, out=out)

    g = sps.gamma(a)

    if out is not None:
        out = sps.gammaincc(a, x, out=out)
        np.multiply(out, g, out=out)
        return out

    res = sps.gammaincc(a, x)
    res *= g
    return res


@overload
def harmonic(n: onp.ToFloat, /, out: None = None) -> float: ...
@overload
def harmonic(n: onp.ToFloatND, /, out: None = None) -> _FloatND: ...
@overload
def harmonic(n: onp.ToFloatND, /, out: _OutT) -> _OutT: ...
def harmonic(
    n: onp.ToFloat | onp.ToFloatND,
    /,
    out: _OutT | None = None,
) -> float | _FloatND | _OutT:
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
    z = np.asanyarray(n) + 1

    if out is not None:
        sps.digamma(z, out=out)
        np.add(out, np.euler_gamma, out=out)
        return out

    hn = sps.digamma(z) + np.euler_gamma
    return hn.item() if np.isscalar(n) else hn


@overload
def norm_sh_jacobi(
    n: lmt.ToOrder0D,
    alpha: onp.ToFloat,
    beta: onp.ToFloat,
) -> float: ...
@overload
def norm_sh_jacobi(
    n: lmt.ToOrderND,
    alpha: onp.ToFloat,
    beta: onp.ToFloat,
) -> _FloatND: ...
def norm_sh_jacobi(
    n: lmt.ToOrder,
    alpha: onp.ToFloat,
    beta: onp.ToFloat,
) -> float | _FloatND:
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
        msg = f"alpha must be > -1, got {alpha}"
        raise ValueError(msg)
    if beta <= -1:
        msg = f"beta must be > -1, got {beta}"
        raise ValueError(msg)

    r = clean_orders(np.asanyarray(n), "n") + 1

    if alpha == beta == 0:
        # shifted Legendre
        c = np.ones(r.shape)
    elif alpha == beta == -1 / 2:
        # shifted Chebychev of the first kind
        c = np.exp(2 * (sps.gammaln(r - 1 / 2) - sps.gammaln(r))) / 2
    elif alpha == beta == 1 / 2:
        # shifted Chebychev of the second kind
        c = np.exp(2 * (sps.gammaln(r + 1 / 2) - sps.gammaln(r + 1))) / 2
    else:
        p, q = r + alpha, r + beta
        c = np.exp(sps.betaln(p, q) - sps.betaln(r, p + beta)) / (p + q - 1)

    return c.item() if np.isscalar(n) else c


@overload
def fourier_jacobi(
    x: onp.ToFloat,
    c: onp.ToFloatND,
    a: onp.ToFloat,
    b: onp.ToFloat,
) -> _Float: ...
@overload
def fourier_jacobi(
    x: onp.ToFloatND,
    c: onp.ToFloatND,
    a: onp.ToFloat,
    b: onp.ToFloat,
) -> _FloatND: ...
def fourier_jacobi(
    x: onp.ToFloat | onp.ToFloatND,
    c: onp.ToFloatND,
    a: onp.ToFloat,
    b: onp.ToFloat,
) -> _Float | _FloatND:
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

            For instance, `[4, 3, 2]` gives
            \( 4 \jacobi{0}{a}{b}{x} + 3 \jacobi{1}{a}{b}{x} + 2 \jacobi{2}{a}{b}{x} \).
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
    c_ = np.array(c, ndmin=1, copy=None)
    if c_.dtype.char in _DTYPE_CHARS:
        c_ = c_.astype(np.float64)

    x_ = np.asanyarray(x)

    if len(c_) == 0:
        return 0.0 if np.isscalar(x) else 0.0 * x_

    # temporarily replace inf's with abs(_) > 1, and track the sign
    if hasinfs := np.any(infs := np.isinf(x_)):
        x_ = np.where(infs, 10 * np.sign(x_), x_)

    # "backwards" recursion (left-reduction)
    # y[k+2] and y[k+1]
    y2, y1 = 0.0, 0.0
    # continue until y[0]
    for k in range(len(c_) - 1, 0, -1):
        # Jacobi recurrence terms
        u, v = a + k, b + k
        w = u + v  # = a + b + 2*k
        # alpha[k]
        p1 = (
            (w + 1)
            * (2 * k * (v - u) + w * (u - v + (w + 2) * x_))
            / (2 * w * (k + 1) * (w - k + 1))
        )
        # beta[k+1]
        q2 = -((u + 1) * (v + 1) * (w + 4) / ((w + 2) * (k + 2) * (w - k + 2)))

        # update the state; "forget" y[k+2]
        y1, y2 = c_[k] + p1 * y1 + q2 * y2, y1

    # results of jacobi polynomial for n=0 and n=1
    f0 = 1
    f1 = (a - float(b) + (a + b + 2) * x_) / 2

    # beta[1]
    q1 = -(a + 1) * (b + 1) * (a + b + 4) / (2 * (a + b + 2) ** 2)

    # Behold! The magic of Clenshaw's algorithm:
    out = c_[0] * f0 + y1 * f1 + y2 * q1 * f0

    # propagation of 'inf' values, ensuring correct sign
    if hasinfs:
        out[infs] = np.inf * np.sign(out[infs] - c_[0] * f0)

    # unpack array iff `x` is scalar; 0-d arrays will pass through
    return out.item() if np.isscalar(x) else out
