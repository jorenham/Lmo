"""Statistical test and tools."""

__all__ = ('normaltest',)

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ._lm import l_ratio


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
