__all__ = 'quad',

import functools
import warnings
from typing import Any, Callable, ParamSpec, cast

import numpy as np
import numpy.typing as npt

P = ParamSpec('P')

@functools.cache
def _leggauss(
    n: int,
    /,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    return cast(
        tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
        np.polynomial.legendre.leggauss(n), # pyright: ignore
    )

def quad(
    f: Callable[[npt.NDArray[np.float_]], npt.NDArray[np.number[Any]]],
    a: float,
    b: float,
    /,
    tol: float = 1e-8,
    rtol: float = 1e-5,
    miniter: int = 2,
    maxiter: int = 100,
) -> tuple[float, float]:
    """
    Gauss-legendre quadrature. Numerically evaluates $\\int_a^b f(u) \\,du$.

    This is a numpy-only implementation that is effectively equivalent to
    [`scipy.interpolate.quadrature`][scipy.interpolate.quadrature].

    Args:
        f: The (vectorized) function to integrate.
        a: Lower integration limit.
        b: Upper integration limit.
        tol: Absolute error tolerance.
        rtol: Relative error tolerance.
        miniter: Minimum number of iterations.
        maxiter: Maximum number of iterations.

    Returns:
        val: The evaluated integral
        err: Absolute difference between the last two fixed-point
            evaluations.

    Examples:
        Calculate $\\int^4_0 x^2 dx$ and compare with an analytic result

        >>> from lmo import integrate
        >>> integrate.quad(lambda x: x**2, 0, 4)
        (21.33333333..., 7.10542735...e-15)
        >>> print(4**3 / 3)  # analytical result
        21.333333333...

    See Also:
        - [`scipy.interpolate.quadrature`][scipy.interpolate.quadrature]

    """
    if np.isinf(a) or np.isinf(b):
        raise ValueError(
            "Gaussian quadrature is only available for finite limits."
        )

    c = (b - a) / 2

    val, err = np.inf, np.inf
    for n in range(miniter + 1, maxiter + 1):
        x, w = _leggauss(n)
        val_n = c * cast(float, np.sum(   # pyright: ignore
            w * f(c * (x + 1) + a),
            axis=-1,
        ))

        err, val = abs(val_n - val), val_n
        if err < tol or err < rtol * abs(val):
            break
    else:
        warnings.warn(f'maxiter ({maxiter}) exceeded with error = {err}')

    return val, err
