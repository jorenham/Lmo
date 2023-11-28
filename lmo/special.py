"""Mathematical "special" functions, extending `scipy.special`."""

__all__ = ('gamma2',)

from typing import cast, overload

import numpy as np
import numpy.typing as npt
import scipy.special as _special  # type: ignore

from .typing import AnyNDArray, AnyScalar


@overload
def gamma2(
    a: float,
    x: AnyScalar,
    out: npt.NDArray[np.float64] | None = ...,
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
