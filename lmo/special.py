"""Mathematical "special" functions, extending `scipy.special`."""

__all__ = ('fpow', 'gamma2', 'harmonic', 'eval_sh_jacobi', 'fourier_jacobi')

from typing import cast, overload

import numpy as np
import numpy.typing as npt
import scipy.special as _special  # type: ignore

from .typing import AnyNDArray, AnyScalar


@overload
def fpow(
    x: AnyScalar,
    n: AnyScalar,
    out: None = ...,
) -> float: ...

@overload
def fpow(
    x: AnyNDArray[np.generic],
    n: npt.ArrayLike,
    out: npt.NDArray[np.float64] | None = ...,
) -> npt.NDArray[np.float64]: ...

@overload
def fpow(
    x: npt.ArrayLike,
    n: AnyNDArray[np.generic],
    out: npt.NDArray[np.float64] | None = ...,
) -> npt.NDArray[np.float64]: ...

@overload
def fpow(
    x: npt.ArrayLike,
    n: npt.ArrayLike,
    out: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

@overload
def fpow(
    x: npt.ArrayLike,
    n: npt.ArrayLike,
    out: npt.NDArray[np.float64] | None = ...,
) -> float | npt.NDArray[np.float64]: ...

def fpow(
    x: npt.ArrayLike,
    n: npt.ArrayLike,
    out: npt.NDArray[np.float64] | None = None,
) -> float | npt.NDArray[np.float64]:
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
    res = cast(
        npt.NDArray[np.float64],
        _special.poch(_x - _n + 1, _n, out=out),  # type: ignore
    )
    if res.ndim == 0 and np.isscalar(x) and np.isscalar(n):
        return res[()]
    return res

@overload
def gamma2(
    a: float,
    x: AnyScalar,
    out: None = ...,
) -> float: ...

@overload
def gamma2(
    a: float,
    x: AnyNDArray[np.generic],
    out: npt.NDArray[np.float64] | None = ...,
) -> npt.NDArray[np.float64]: ...

@overload
def gamma2(
    a: float,
    x: npt.ArrayLike,
    out: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

@overload
def gamma2(
    a: float,
    x: npt.ArrayLike,
    out: npt.NDArray[np.float64] | None = ...,
) -> float | npt.NDArray[np.float64]: ...

def gamma2(
    a: float,
    x: npt.ArrayLike,
    out: npt.NDArray[np.float64] | None = None,
) -> float | npt.NDArray[np.float64]:
    r"""
    Incomplete (upper) gamma function.

    It is defined as

    \[
        \Gamma(a,\ x) = \int_x^\infty t^{a-1} e^{-t} \mathrm{d}t
    \]

    for \( a \ge 0 \) and \( x \ge 0 \).

    Args:
        a: Non-negative scalar.
        x: Non-negative array-like.
        out: Optional output array for the results.

    Returns:
        out: Scalar of array with the values of the incomplete gamma function.

    See Also:
        - [`scipy.special.gammaincc`][scipy.special.gammaincc] for the
          regularized gamma function \( Q(a,\ x) \).
    """
    if a == 0:
        return cast(
            float | npt.NDArray[np.float64],
            _special.exp1(x, out=out),  # type: ignore
        )

    res = cast(
        float | npt.NDArray[np.float64],
        _special.gammaincc(a, x, out=out),  # type: ignore
    )
    res *= cast(float, _special.gamma(a))  # type: ignore
    return res


def harmonic(
    n: npt.ArrayLike,
    /,
    out: npt.NDArray[np.float64] | npt.NDArray[np.complex128] | None = None,
) -> float | complex | npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
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
        4.3267...
        >>> harmonic(np.pi)
        1.8727...
        >>> harmonic(-1 / 12)
        -0.1461...
        >>> harmonic(1 - 1j)
        (1.1718...-0.5766...j)

    Args:
        n: Real- or complex- valued parameter, as array-like or scalar.
        out: Optional real or complex output array for the results.

    Returns:
        out: Array or scalar with the value(s) of the function.

    See Also:
        - [Harmonic number - Wikipedia
        ](https://wikipedia.org/wiki/Harmonic_number)
    """
    _n = np.asanyarray(n)

    _out = cast(
        npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        _special.digamma(_n + 1, out),  # type: ignore
    )
    _out += np.euler_gamma

    return _out[()] if np.isscalar(n) else _out

@overload
def eval_sh_jacobi(
    n: int,
    alpha: float,
    beta: float,
    x: AnyScalar,
    out: None = ...,
) -> float: ...

@overload
def eval_sh_jacobi(
    n: int,
    alpha: float,
    beta: float,
    x: AnyNDArray[np.generic],
    out: npt.NDArray[np.float64] | None = ...,
) -> npt.NDArray[np.float64]: ...

@overload
def eval_sh_jacobi(
    n: int,
    alpha: float,
    beta: float,
    x: npt.ArrayLike,
    out: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

@overload
def eval_sh_jacobi(
    n: int,
    alpha: float,
    beta: float,
    x: npt.ArrayLike,
    out: npt.NDArray[np.float64] | None = ...,
) -> float | npt.NDArray[np.float64]: ...

def eval_sh_jacobi(
    n: int,
    alpha: float,
    beta: float,
    x: npt.ArrayLike,
    out: npt.NDArray[np.float64] | None = None,
) -> float | npt.NDArray[np.float64]:
    r"""
    Evaluate the (correct) shifted Jacobi Polynomial
    \( \shjacobi{n}{\alpha}{\beta}{u} \) on \( x \in [0, 1] \), i.e.
    the Jacobi Polynomial with mapped argument as \( x \mapsto 2x - 1 \).

    It is well known that the "shifted Legendre" polynomial is the Legendre
    polynomial with \( 2x - 1 \) as mapped argument, which is correctly
    implemented in
    [`scipy.special.eval_sh_legendre`][scipy.special.eval_sh_legendre].
    Any textbook on orthogonal
    polynomials will tell you that the generalization of Legendre are the
    Jacobi polynomials. Hence, the only valid interpretation of the
    "shifted Jacobi polynomials", should be the analogue (homomorphism) of
    the shifted Legendre polynomials.

    However, [`scipy.special.eval_sh_jacobi`][scipy.special.eval_sh_jacobi]
    are **not** the shifted Jacobi polynomials!.
    Instead, the method evaluates the *generalized Gegenbauer polynomials*.
    The Jacobi-, and Legendre polynomials are denoted
    with a "P", which stands for "polynomial". In the `eval_sh_jacobi`
    docstring, the notation \( G_n^{p,q} \) is used. Clearly, the "G" stands
    for "Gegenbauer".
    See [scipy/scipy#18988](https://github.com/scipy/scipy/issues/18988) for
    the relevant issue.
    """
    if alpha == beta == 0:
        return _special.eval_sh_legendre(n, x, out)  # type: ignore

    y = 2 * np.asanyarray(x) - 1
    return _special.eval_jacobi(n, alpha, beta, y, out)  # type: ignore


def fourier_jacobi(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    a: float,
    b: float,
) -> float | npt.NDArray[np.float64]:
    """
    Evaluate the Fourier-Jacobi series, using the Clenshaw summation
    algorithm.

    See Also:
        - [Generalized Fourier series - Wikipedia](
        https://wikipedia.org/wiki/Generalized_Fourier_series)
        - [Clenshaw Recurrence Formula - Wolfram MathWorld](
        https://mathworld.wolfram.com/ClenshawRecurrenceFormula.html)
        - [Jacobi Polynomial - Worlfram Mathworld](
        https://mathworld.wolfram.com/JacobiPolynomial.html)
    """
    _c = np.array(c, ndmin=1, copy=False)
    if _c.dtype.char in '?bBhHiIlLqQpP':
        _c = _c.astype(np.float64)

    _x = np.asanyarray(x)

    if len(_c) == 0:
        return 0. * _x


    # "backwards" recursion (left-reduction)
    # y[k+2] and y[k+1]
    y2, y1 = 0, 0
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
    return _c[0] * f0 + y1 * f1 + y2 * q1 * f0
