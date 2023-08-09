"""Statistical test and tools."""

__all__ = ('normaltest', 'l_ratio_bounds')

from collections.abc import Sequence
from typing import Any, NamedTuple, TypeVar, overload

import numpy as np
import numpy.typing as npt

from ._lm import l_ratio
from .typing import AnyInt

T = TypeVar('T', bound=np.floating[Any])


class NormaltestResult(NamedTuple):
    statistic: float | npt.NDArray[np.float_]
    pvalue: float | npt.NDArray[np.float_]


def normaltest(
    a: npt.ArrayLike,
    /,
    *,
    axis: int | None = None,
) -> NormaltestResult:
    r"""
    Test the null hypothesis that a sample comes from a normal distribution.
    Based on the Harri & Coble (2011) test, and includes Hosking's correction.

    Args:
        a: The array-like data.
        axis: Axis along which to compute the test.

    Returns:
        statistic: The $\tau^2_{3, 4}$ test statistic.
        pvalue: A 2-sided chi squared probability for the hypothesis test.

    Examples:
        Compare the testing power with
        [`scipy.stats.normaltest`][scipy.stats.normaltest] given 10.000 samples
        from a contaminated normal distribution.

        >>> import numpy as np
        >>> from lmo.diagnostic import normaltest
        >>> from scipy.stats import normaltest as normaltest_scipy
        >>> rng = np.random.default_rng(12345)
        >>> n = 10_000
        >>> x = 0.9 * rng.normal(0, 1, n) + 0.1 * rng.normal(0, 9, n)
        >>> normaltest(x)[1]
        0.04806618...
        >>> normaltest_scipy(x)[1]
        0.08435627...

    References:
        [A. Harri & K.H. Coble (2011) - Normality testing: Two new tests
        using L-moments](https://doi.org/10.1080/02664763.2010.498508)
    """
    x = np.asanyarray(a)

    # sample size
    n = x.size if axis is None else x.shape[axis]

    # L-skew and L-kurtosis
    t3, t4 = l_ratio(a, [3, 4], [2, 2], axis=axis)

    # theoretical L-skew and L-kurtosis of the normal distribution (for all
    # loc/mu and scale/sigma)
    tau3, tau4 = 0.0, 30/np.pi * np.arctan(np.sqrt(2)) - 9

    z3 = (t3 - tau3) / np.sqrt(
        0.1866 / n
        + (np.sqrt(0.8000) / n)**2,
    )
    z4 = (t4 - tau4) / np.sqrt(
        0.0883 / n
        + (np.sqrt(0.6800) / n)**2
        + (np.cbrt(4.9000) / n)**3,
    )

    k2 = z3**2 + z4**2

    # chi2(k=2) survival function (sf)
    p_value = np.exp(-k2 / 2)

    return NormaltestResult(k2, p_value)


@overload
def l_ratio_bounds(
    r: AnyInt,
    /,
    trim: tuple[float, float] = ...,
    *,
    dtype: np.dtype[np.float_] = ...,
) -> np.float_:
    ...

@overload
def l_ratio_bounds(
    r: AnyInt,
    /,
    trim: tuple[float, float] = ...,
    *,
    dtype: np.dtype[T] | type[T],
) -> T:
    ...


@overload
def l_ratio_bounds(
    r: npt.NDArray[Any] | Sequence[Any],
    /,
    trim: tuple[float, float] = ...,
    *,
    dtype: np.dtype[np.float_] = ...,
) -> npt.NDArray[np.float_]:
    ...

@overload
def l_ratio_bounds(
    r: npt.NDArray[Any] | Sequence[Any],
    /,
    trim: tuple[float, float] = ...,
    *,
    dtype: np.dtype[T] | type[T],
) -> npt.NDArray[T]:
    ...


@overload
def l_ratio_bounds(
    r: npt.ArrayLike,
    /,
    trim: tuple[float, float] = ...,
    *,
    dtype: np.dtype[np.float_] = ...,
) -> np.float_ | npt.NDArray[np.float_]:
    ...

@overload
def l_ratio_bounds(
    r: npt.ArrayLike,
    /,
    trim: tuple[float, float] = ...,
    *,
    dtype: np.dtype[T] | type[T],
) -> T | npt.NDArray[T]:
    ...


def l_ratio_bounds(
    r: npt.ArrayLike,
    /,
    trim: tuple[float, float] = (0, 0),
    *,
    dtype: np.dtype[T] | type[T] = np.float_,
) -> T | npt.NDArray[T]:
    r"""
    Upper bounds on the absolute L-moment ratio's. It is based on the
    bounds introduced by Hosking in 2007, but generalized to allow for
    fractional trimming (i.e. $(s, t) \in \mathbb{R}^2_+$), and reformulated
    as a recursive definition.

    $$
    |\tau_r^{(s, t)}| \le
        \frac{2}{r} \frac
        {\Gamma (s \wedge t + 2) \Gamma (s + t + r + 1)}
        {\Gamma (s \wedge t + r) \Gamma (s + t + 3)}
    $$

    Here, $s \wedge t$ denotes the smallest trim-length, i.e. `min(s, t)`.

    Notes:
        Without trim (default), the upper bound is $1$ for all $r \neq 1$.

    Args:
        r: Order of the L-moment ratio(s), as positive integer scalar or
            array-like.
        trim: Tuple of left- and right- trim-lengths, matching those of the
            relevant L-moment ratio's.
        dtype: Floating type to use.

    Returns:
        Array or scalar with shape like $r$.

    See Also:
        - [`l_ratio`][lmo.l_ratio]
        - [`l_ratio_se`][lmo.l_ratio_se]

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
        L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    """
    _rs = np.asarray_chkfinite(r, dtype=int)[()]
    if np.any(_rs < 0):
        msg = 'expected r >= 0'
        raise ValueError(msg)

    _n = np.max(_rs) + 1
    if _n > 9000:
        msg = f"max(r) = {_n - 1}; it's over 9000!"
        raise ValueError(msg)

    # the zeroth L-ratio is 1/1, the 2nd L-ratio is L-scale / L-scale
    out = np.ones(_n, dtype=dtype)

    if _n > 1:
        # `L-loc / L-scale (= 1 / CV)` is unbounded
        out[1] = np.inf

    s, t = np.asarray_chkfinite(trim)
    if s < 0 or t < 0:
        msg = f'trim lengths must be positive, got {trim}'
        raise ValueError(msg)

    if _n > 3 and s and t:
        # if not trimmed, then the bounds are simply 1
        p, q = s + t, min(s, t)
        for _r in range(3, _n):
            out[_r] = out[_r - 1] * (1 + p / _r) / (1 + q / (_r - 1))

    return out[_rs]
