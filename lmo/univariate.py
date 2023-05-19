"""
Estimators of the sample L-moments, and derived summary statistics.

According to [Wikipedia](https://wikipedia.org/wiki/L-moment):

> L-moments are far more meaningful when dealing with outliers in data
> than conventional moments.

Note that L-moments are robust to outliers, but not resistant to extreme
values.

Often the Method of L-moment (LMM) outperforms the conventional method of
moments (MM) and maximum likelihood estimation (MLE), e.g. ftting of the
``scipy.stats.genextreme`` (generalized extreme value, GED) distribution.


See Also:
  * [J.R.M. Hosking (1990) - L-Moments: Analysis and Estimation of
    Distributions Using Linear Combinations of
    Order Statistics](https://jstor.org/stable/2345653)
  * [E. Elmamir & A. Seheult (2003) -
    Trimmed L-moments](https://doi.org/10.1016/S0167-9473(02)00250-5)
  * [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
    L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

"""

__all__ = (
    'l_moment', 'l_ratio', 'l_loc', 'l_scale', 'l_skew', 'l_kurt',
    'tl_moment', 'tl_ratio', 'tl_loc', 'tl_scale', 'tl_skew', 'tl_kurt',
)

from typing import Any

import numpy as np

from .typing import AnyTensor, ScalarOrArray, SortKind, Trimming
from .weights import tl_weights


def tl_moment(
    a: AnyTensor,
    r: int,
    /,
    trim: Trimming = 1,
    *,
    axis: int | None = None,
    sort: SortKind = None,
) -> ScalarOrArray[np.float_]:
    """
    Estimate the $r$-th sample TL-moment, $\\lambda_{r}^{(t_1, t_2)}$, for
    left and right trim lengths $t_1$ and $t_2$.

    Parameters:
        a (array_like):
            Array containing numbers whose TL-moment is desired. If `a` is not
            an array, a conversion is attempted.
        r (int):
            The order of the TL-moment. Some special cases cases include

            - `0`: Like the zeroth moment, the zeroth TL-moment  is always `1`.
            - `1`: The TL-location, the analogue of the mean. See
                [`tl_loc`][lmo.tl_loc].
            - `2`: The TL-scale, analogous to the standard deviation. See
                [`tl_scale`][lmo.tl_scale].

        trim (int | tuple[int, int]):
            Amount of samples to trim as either

            - `t: int` for symmetric trimming, equivalent to `(t, t)`.
            - `(t1: int, t2: int)` for asymmetric trimming, or

            If `0` is passed, the L-moment is returned.

    Other parameters:
        axis (int?):
            Axis along wich to calculate the TL-moments.
            If `None` (default), all samples in the array will be used.
        sort ('quicksort' | 'mergesort' | 'heapsort' | 'stable'):
            Sorting algorithm, see [`numpy.sort`](
            https://numpy.org/doc/stable/reference/generated/numpy.sort).

    Returns:
        Scalar or array; the $r$-th TL-moment(s).

    """
    x = np.sort(np.asanyarray(a), axis=axis, kind=sort)
    n = x.shape[axis or 0]

    if r == 0:
        # zeroth (TL-)moment is 1
        return np.ones(x.size // n) if x.ndim > 1 else np.float_(1)

    w = tl_weights(n, r, trim)

    _axis = (axis or 0) % x.ndim
    assert _axis >= 0

    if _axis == 0:
        # apply along the first axis
        return w @ x

    if x.ndim == 2 or _axis == x.ndim - 1:
        # apply along the last axis
        return x @ w

    return np.apply_along_axis(np.inner, _axis, x, w)


def tl_ratio(
    a: AnyTensor,
    r: int,
    /,
    k: int = 2,
    trim: Trimming = 1,
    *,
    axis: int | None = None,
    sort: SortKind = None,
) -> ScalarOrArray[np.float_]:
    """
    Ratio of the r-th and k-th (2nd by default) sample TL-moments:

    $$
    \\tau_{r, k}^{(t_1, t_2)} = \\frac{
        \\lambda_{r}^{(t_1, t_2)}
    }{
        \\lambda_{k}^{(t_1, t_2)}
    }
    $$

    By default, $k = 2$ and $t_1 = t_2 = 1$ are used, i.e.
    $\\tau_{r, 2}^{(1, 1)}$, or $\\tau_r^{(1)}$ for short.

    """
    x = np.sort(np.asanyarray(a), axis=axis, kind=sort)

    l_r = tl_moment(x, r, trim, axis=axis)

    if k == 0:
        return l_r
    if k == r:
        return np.ones_like(l_r)[()]

    l_k = l_r if k == r else tl_moment(x, k, trim, axis=axis)

    # i.e. `x / 0 = 0 if x == 0 else np.nan`
    return np.divide(
        l_r,
        l_k,
        out=np.where(l_r == 0, 0.0, np.nan),
        where=l_k != 0
    )[()]  # [()] converts any 0-dimensional arrays to scalar


def tl_loc(
    a: AnyTensor,
    /,
    trim: Trimming = 1,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    Sample estimator of the TL-location, $\\lambda_1^{(t_1, t_2)}$; the first
    TL-moment.

    See Also:
         [lmo.tl_moment][lmo.univariate.tl_moment]
    """
    return tl_moment(a, 1, trim, **kwargs)


def tl_scale(
    a: AnyTensor,
    /,
    trim: Trimming = 1,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    Sample TL-scale estimator, $\\lambda_2^{(t_1, t_2)}$, the second TL-moment.
    A robust alternative of the sample standard deviation.
    """
    return tl_moment(a, 2, trim, **kwargs)


def tl_skew(
    a: AnyTensor,
    /,
    trim: Trimming = 1,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    TL-skewness coefficient, $\\tau_3^{(t_1, t_2)}$; the 3rd sample TL-moment
    ratio.
    """
    return tl_ratio(a, 3, trim=trim, **kwargs)


def tl_kurt(
    a: AnyTensor,
    /,
    trim: Trimming = 1,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    TL-kurtosis coefficient, $\\tau_4^{(t_1, t_2)}$; the 4th sample TL-moment
    ratio.
    """
    return tl_ratio(a, 4, trim=trim, **kwargs)


# L-moment aliasses

def l_moment(
    a: AnyTensor,
    r: int,
    /,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    The $r$-th sample L-moment, $\\lambda_r$.
    Alias of [`lmo.tl_moment(..., trim=0)`][lmo.univariate.tl_moment].

    See Also:
        - [J.R.M. Hosking (1990)](https://jstor.org/stable/2345653)
        - [L-moment - Wikipedia](https://wikipedia.org/wiki/L-moment)

    """
    return tl_moment(a, r, trim=0, **kwargs)


def l_ratio(
    a: AnyTensor,
    r: int,
    /,
    k: int = 2,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    Ratio of the r-th and k-th (2nd by default) sample L-moments:

    $$
    \\tau_{r, k} = \\frac{\\lambda_{r}}{\\lambda_{k}}
    $$

    Alias of [`lmo.tl_ratio(..., trim=0)`][lmo.univariate.tl_ratio].

    Notes:
        Tthe L-moment ratio's are bounded within the interval $[-1, 1)$.

    """
    return tl_ratio(a, r, k, trim=0, **kwargs)


def l_loc(a: AnyTensor, /, **kwargs: Any) -> ScalarOrArray[np.float_]:
    """
    L-location: the first sample L-moment.
    Equivalent to [`lmo.tl_loc(a, 0, **kwargs)`][lmo.univariate.tl_loc].

    Notes:
        The L-location is equivalent to the (arithmetic) sample mean.

    """
    return l_moment(a, 1, **kwargs)


def l_scale(a: AnyTensor, /, **kwargs: Any) -> ScalarOrArray[np.float_]:
    """
    L-scale: the second L-moment.
    Equivalent to [`lmo.tl_scale(a, 0, **kwargs)`][lmo.univariate.tl_scale].

    Notes:
        The L-scale is equivalent to half the [mean absolute difference](
        https://wikipedia.org/wiki/Mean_absolute_difference).

    """
    return l_moment(a, 2, **kwargs)


def l_skew(a: AnyTensor, /, **kwargs: Any) -> ScalarOrArray[np.float_]:
    """
    L-skewness coefficient; the 3rd sample L-moment ratio.
    Equivalent to [`lmo.tl_skew(a, 0, **kwargs)`][lmo.univariate.tl_skew].
    """
    return l_ratio(a, 3, **kwargs)


def l_kurt(a: AnyTensor, /, **kwargs: Any) -> ScalarOrArray[np.float_]:
    """
    L-kurtosis coefficient; the 4th sample L-moment ratio.
    Equivalent to [`lmo.tl_kurt(a, 0, **kwargs)`][lmo.univariate.tl_kurt].

    Notes:
        The L-kurtosis $\\tau_4$ lies within the interval
        $[-\\frac{1}{4}, 1)$, and by the L-skewness $\\tau_3$ as
        $5 \\tau_3^2 - 1 \\le 4 \\tau_4$.

    """
    return l_ratio(a, 4, **kwargs)
