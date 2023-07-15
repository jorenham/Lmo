"""
Theoretical L-moments of known and unknown probability distributions.
"""

__all__ = (
    'l_moment_from_cdf',
    'l_moment_from_ppf',
)

from math import exp, lgamma
from typing import Any, Callable, cast, overload
import functools
import warnings

import numpy as np
import numpy.typing as npt

from ._utils import clean_order
from .linalg import sh_jacobi
from .typing import AnyFloat, AnyInt, IntVector


def _l_moment_const(r: int, s: float, t: float, k: int) -> float:
    return exp(
        lgamma(r - k)
        + lgamma(r + s + t + 1)
        - lgamma(r + s)
        - lgamma(r + t)
    ) / r if r > k else 1.


@overload
def l_moment_from_cdf(
    cdf: Callable[[float], float],
    r: AnyInt,
    /,
    trim: tuple[AnyFloat, AnyFloat] = ...,
    support: tuple[AnyFloat, AnyFloat] = ...,
) -> float:
    ...


@overload
def l_moment_from_cdf(
    cdf: Callable[[float], float],
    r: IntVector,
    /,
    trim: tuple[AnyFloat, AnyFloat] = ...,
    support: tuple[AnyFloat, AnyFloat] = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_cdf(
    cdf: Callable[[float], float],
    r: AnyInt | IntVector,
    /,
    trim: tuple[AnyFloat, AnyFloat] = (0, 0),
    support: tuple[AnyFloat, AnyFloat] = (-np.inf, np.inf),
) -> float | npt.NDArray[np.float_]:
    # TODO: docstring

    _r = np.asanyarray(r)
    if not np.issubdtype(_r.dtype, np.integer):
        raise TypeError(f'r must be integer-valued, got {_r.dtype.str!r}')
    if _r.size == 0:
        raise ValueError('no r provided')
    if np.any(_r < 0):
        raise ValueError('r must be non-negative')

    s, t = np.asanyarray(trim)

    r_max = clean_order(np.max(_r))
    r_vals, r_idxs = np.unique(_r, return_inverse=True)
    assert r_vals.ndim == 1

    # caching F(x) function only makes sense for multiple quad calls
    F = functools.cache(cdf) if np.count_nonzero(r_vals) > 1 else cdf

    # shifted Jacobi polynomial coefficients
    j = sh_jacobi(r_max - 1, t + 1, s + 1)

    # lazy import (don't worry; python imports are cached)
    from scipy.integrate import quad, IntegrationWarning  # type: ignore

    l_r = np.empty(r_vals.shape)
    for i, r_val in np.ndenumerate(r_vals):
        if r_val == 0:
            # zeroth l-moment is always 1
            l_r[i] = 1
            continue

        if r_val == 1:
            if s == t == 0:
                def integrand(x: float, *args: Any) -> float:
                    # equivalent to E[X], i.e. the mean
                    return (x >= 0) - F(x, *args)
            else:
                from scipy.special import betainc  # type: ignore

                def integrand(x: float, *args: Any) -> float:
                    # equivalent to E[X_{s+1 : s+t+1}]
                    # see Wiley (2003) eq. 2.1.5
                    p = F(x, *args)
                    return (x >= 0) - betainc(s + 1, t + 1, p)  # type: ignore

        else:
            # prepare the powers to use for evaluating the polynomial
            k = cast(npt.NDArray[np.int_], np.arange(r_val - 1))
            # grab the non-zero jacobi polynomial coefficients for k=r-1
            j_k = j[r_val - 2, :r_val - 1]

            def integrand(x: float, *args: Any) -> float:
                # evaluate the jacobi polynomial for p at r-1 with (t, s)
                # and multiply by the weight function
                p = F(x, *args)
                return p ** (s + 1) * (1 - p) ** (t + 1) * (j_k @ p**k)

        # numerical integration
        quad_val, _, _, *quad_tail = cast(
            tuple[float, float, dict[str, Any]]
            | tuple[float, float, dict[str, Any], str],
            quad(integrand, *support, full_output=True)
        )

        if quad_tail:
            quad_msg = quad_tail[0]
            warnings.warn(
                f"'scipy.integrate.quad' failed: \n{quad_msg}",
                cast(type[UserWarning], IntegrationWarning),
                stacklevel=2
            )
            l_r[i] = np.nan
            continue

        l_r[i] = _l_moment_const(r_val, s, t, 1) * quad_val

    return np.round(l_r, 12)[r_idxs if _r.ndim > 0 else 0] + .0


@overload
def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: AnyInt,
    /,
    trim: tuple[AnyFloat, AnyFloat] = ...,
    support: tuple[AnyFloat, AnyFloat] = ...,
) -> float:
    ...


@overload
def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: IntVector,
    /,
    trim: tuple[AnyFloat, AnyFloat] = ...,
    support: tuple[AnyFloat, AnyFloat] = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: AnyInt | IntVector,
    /,
    trim: tuple[AnyFloat, AnyFloat] = (0, 0),
    support: tuple[AnyFloat, AnyFloat] = (0, 1),
) -> float | npt.NDArray[np.float_]:
    # TODO: docstring

    _r = np.asanyarray(r)
    if not np.issubdtype(_r.dtype, np.integer):
        raise TypeError(f'r must be integer-valued, got {_r.dtype.str!r}')
    if _r.size == 0:
        raise ValueError('no r provided')
    if np.any(_r < 0):
        raise ValueError('r must be non-negative')

    s, t = np.asanyarray(trim)

    r_max = clean_order(np.max(_r))
    r_vals, r_idxs = np.unique(_r, return_inverse=True)
    assert r_vals.ndim == 1

    def _w(p: float, *args: Any) -> float:
        return p**s * (1 - p)**t * ppf(p, *args)

    # caching the weight function only makes sense for multiple quad calls
    w = functools.cache(_w) if len(r_vals) > 1 else _w

    # shifted Jacobi polynomial coefficients
    j = sh_jacobi(r_max, t, s)

    # lazy import (don't worry; python imports are cached)
    from scipy.integrate import quad, IntegrationWarning  # type: ignore

    l_r = np.empty(r_vals.shape)
    for i, r_val in np.ndenumerate(r_vals):
        if r_val == 0:
            # zeroth l-moment is always 1
            l_r[i] = 1
            continue

        # prepare the powers to use for evaluating the polynomial
        k = cast(npt.NDArray[np.int_], np.arange(r_val))
        # grab the non-zero jacobi polynomial coefficients for k=r-1
        j_k = j[r_val - 1, :r_val]

        def integrand(p: float) -> float:
            # evaluate the jacobi polynomial for p at r-1 with (t, s)
            # and multiply by the weight function
            return w(p) * (j_k @ p**k)  # type: ignore

        # numerical integration
        quad_val, _, _, *quad_tail = cast(
            tuple[float, float, dict[str, Any]]
            | tuple[float, float, dict[str, Any], str],
            quad(integrand, *support, full_output=True)
        )
        if quad_tail:
            quad_msg = quad_tail[0]
            warnings.warn(
                f"'scipy.integrate.quad' failed: \n{quad_msg}",
                cast(type[UserWarning], IntegrationWarning),
                stacklevel=2
            )
            l_r[i] = np.nan
            continue

        l_r[i] = _l_moment_const(r_val, s, t, 0) * quad_val

    return np.round(l_r, 12)[r_idxs if _r.ndim > 0 else 0] + .0
