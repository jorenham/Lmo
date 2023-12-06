"""Mathematical "special" functions, extending `scipy.special`."""

__all__ = ('fpow', 'gamma2', 'harmonic')

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
