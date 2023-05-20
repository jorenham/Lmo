"""
Estimators of the sample L- and TL-moments, and related summary statistics.
"""

__all__ = (
    'l_moment',
    'l_ratio',
    'l_loc',
    'l_scale',
    'l_skew',
    'l_kurt',

    'tl_moment',
    'tl_ratio',
    'tl_loc',
    'tl_scale',
    'tl_skew',
    'tl_kurt',
)

from typing import Any

import numpy as np
import numpy.typing as npt

from .typing import AnyTensor, ScalarOrArray, SortKind, Trimming
from .weights import tl_weights, reweight


def tl_moment(
    a: AnyTensor,
    r: int,
    /,
    trim: Trimming = 1,
    axis: int | None = None,
    *,
    weights: AnyTensor | None = None,
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

        axis (int?):
            Axis along wich to calculate the TL-moments.
            If `None` (default), all samples in the array will be used.

        weights (array_like, optional):
            An array of weights associated with the values in `a`. Each value
            in `a` contributes to the average according to its associated
            weight.
            The weights array can either be 1-D (in which case its length must
            be the size of a along the given axis) or of the same shape as `a`.
            If `weights=None`, then all data in `a` are assumed to have a
            weight equal to one.

            All `weights` must be `>=0`, and the sum must be nonzero.

            The algorithm is similar to that of the weighted median. See
            [`lmo.weights.reweight`][lmo.weights.reweight] for details.

    Other parameters:
        sort ('quick' | 'heap' | 'stable' | 'merge'):
            Sorting algorithm, see [`numpy.sort`](
            https://numpy.org/doc/stable/reference/generated/numpy.sort).

    Returns:
        Scalar or array; the $r$-th TL-moment(s).

    See Also:
        - [E. Elmamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    """
    x = np.asanyarray(a)
    w_x = None if weights is None else np.asanyarray(weights)

    # ensure that the samples are along the first axis (shape standardization)
    # this way we don't have the bother with the specific axis from here on
    if axis is None:
        x = x.ravel()
    if x.ndim > 1 and w_x is not None:
        x, w_x = np.broadcast_arrays(x, w_x)
    if axis and (ax := axis % x.ndim):
        x = np.moveaxis(x, ax, 0)
        if w_x is not None:
            w_x = np.moveaxis(x, ax, 0)

    n = len(x)

    if r == 0:
        # zeroth (TL-)moment is always 1
        # the _[()] ensures that those annoying 0d "arrays" become scalars
        return np.ones(x.shape[1:])[()]

    # calculate the TL-weight vector (it anoly depends on n and r, not on the
    # amount of variables)
    w_r = tl_weights(n, r, trim)

    def _apply_weights_1d(
        _x: npt.NDArray[np.floating[Any]],
        _w_x: npt.NDArray[Any] | None = None
    ) -> npt.NDArray[np.float_]:
        assert _x.ndim == 1
        assert _w_x is None or _w_x.shape == _x.shape

        if _w_x is None:
            _x_k, _w = np.sort(_x, kind=sort), w_r
        else:
            i_k = np.argsort(_x, kind=sort)
            _x_k, _w = _x[i_k], reweight(w_r, _w_x[i_k])

        return _x_k @ _w

    if x.ndim == 1:
        return _apply_weights_1d(x, w_x)

    if w_x is None:
        return np.apply_along_axis(_apply_weights_1d, 0, x)

    # manual version of numpy.apply_along_axis, which handles only one array
    out = np.empty(x.shape[1:], dtype=np.result_type(x, w_r, w_x))
    for jj in np.ndindex(*out.shape):
        out[jj] = _apply_weights_1d(x[jj], w_x[jj])
    return out


def tl_ratio(
    a: AnyTensor,
    r: int,
    /,
    k: int = 2,
    trim: Trimming = 1,
    axis: int | None = None,
    **kwargs: Any,
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
    x = np.asanyarray(a)

    l_r = tl_moment(x, r, trim, axis=axis, **kwargs)

    if k == 0:
        return l_r
    if k == r:
        return np.ones_like(l_r)[()]

    l_k = l_r if k == r else tl_moment(x, k, trim, axis=axis, **kwargs)

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
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    Sample estimator of the TL-location, $\\lambda_1^{(t_1, t_2)}$; the first
    TL-moment.

    See Also:
         [lmo.tl_moment][lmo.univariate.tl_moment]
    """
    return tl_moment(a, 1, trim, axis=axis, **kwargs)


def tl_scale(
    a: AnyTensor,
    /,
    trim: Trimming = 1,
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    Sample TL-scale estimator, $\\lambda_2^{(t_1, t_2)}$, the second TL-moment.
    A robust alternative of the sample standard deviation.
    """
    return tl_moment(a, 2, trim, axis=axis, **kwargs)


def tl_skew(
    a: AnyTensor,
    /,
    trim: Trimming = 1,
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    TL-skewness coefficient, $\\tau_3^{(t_1, t_2)}$; the 3rd sample TL-moment
    ratio.
    """
    return tl_ratio(a, 3, trim=trim, axis=axis, **kwargs)


def tl_kurt(
    a: AnyTensor,
    /,
    trim: Trimming = 1,
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    TL-kurtosis coefficient, $\\tau_4^{(t_1, t_2)}$; the 4th sample TL-moment
    ratio.
    """
    return tl_ratio(a, 4, trim=trim, axis=axis, **kwargs)


# L-moment aliasses

def l_moment(
    a: AnyTensor,
    r: int,
    /,
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    The $r$-th sample L-moment, $\\lambda_r$.
    Alias of [`lmo.tl_moment(..., trim=0)`][lmo.univariate.tl_moment].

    According to [Wikipedia](https://wikipedia.org/wiki/L-moment):

    > L-moments are far more meaningful when dealing with outliers in data
    > than conventional moments.

    Note that L-moments are robust to outliers, but not resistant to extreme
    values.

    Often the Method of L-moment (LMM) outperforms the conventional method of
    moments (MM) and maximum likelihood estimation (MLE), e.g. ftting of the
    ``scipy.stats.genextreme`` (generalized extreme value, GED) distribution.


    See Also:
        - [J.R.M. Hosking (1990)](https://jstor.org/stable/2345653)
        - [L-moment - Wikipedia](https://wikipedia.org/wiki/L-moment)

    """
    return tl_moment(a, r, trim=0, axis=axis, **kwargs)


def l_ratio(
    a: AnyTensor,
    r: int,
    /,
    k: int = 2,
    axis: int | None = None,
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
    return tl_ratio(a, r, k, trim=0, axis=axis, **kwargs)


def l_loc(
    a: AnyTensor,
    /,
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    L-location: the first sample L-moment.
    Equivalent to [`lmo.tl_loc(a, 0, **kwargs)`][lmo.univariate.tl_loc].

    Notes:
        The L-location is equivalent to the (arithmetic) sample mean.

    """
    return l_moment(a, 1, axis=axis, **kwargs)


def l_scale(
    a: AnyTensor,
    /,
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    L-scale: the second L-moment.
    Equivalent to [`lmo.tl_scale(a, 0, **kwargs)`][lmo.univariate.tl_scale].

    Notes:
        The L-scale is equivalent to half the [mean absolute difference](
        https://wikipedia.org/wiki/Mean_absolute_difference).

    """
    return l_moment(a, 2, axis=axis, **kwargs)


def l_skew(
    a: AnyTensor,
    /,
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    L-skewness coefficient; the 3rd sample L-moment ratio.
    Equivalent to [`lmo.tl_skew(a, 0, **kwargs)`][lmo.univariate.tl_skew].
    """
    return l_ratio(a, 3, axis=axis, **kwargs)


def l_kurt(
    a: AnyTensor,
    /,
    axis: int | None = None,
    **kwargs: Any,
) -> ScalarOrArray[np.float_]:
    """
    L-kurtosis coefficient; the 4th sample L-moment ratio.
    Equivalent to [`lmo.tl_kurt(a, 0, **kwargs)`][lmo.univariate.tl_kurt].

    Notes:
        The L-kurtosis $\\tau_4$ lies within the interval
        $[-\\frac{1}{4}, 1)$, and by the L-skewness $\\tau_3$ as
        $5 \\tau_3^2 - 1 \\le 4 \\tau_4$.

    """
    return l_ratio(a, 4, axis=axis, **kwargs)
