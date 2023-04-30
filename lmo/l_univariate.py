"""
Estimators of the sample L-moments, and derived summary statistics.

According to wikipedia:

    L-moments are far more meaningful when dealing with outliers in data
    than conventional moments.

Note that L-moments are robust to outliers, but not resistant to extreme
values.

Often the Method of L-moment (LMM) outperforms the conventional method of
moments (MM) and maximum likelihood estimation (MLE), e.g. ftting of the
``scipy.stats.genextreme`` (generalized extreme value, GED) distribution.


See Also:
    * `L-moment - Wikipedia <https://wikipedia.org/wiki/L-moment>`_
    * `J.R.M. Hosking (1990) <https://jstor.org/stable/2345653>`_

"""

__all__ = 'l_moments', 'l_moment', 'l_loc', 'l_scale', 'l_skew', 'l_kurt',


import numpy as np
import numpy.typing as npt

from ._l_stats import l_weights
from ._typing import SortKind


def l_moments(
    a: npt.ArrayLike,
    k_max: int,
    /,
    axis: int | None = -1,
    *,
    sort_kind: SortKind | None = None,
) -> npt.NDArray[np.float_]:
    """
    Estimate the [0, 1, ..., k_max]-th sample L-moments, where the 0th is 1,
    the first the arithmetic mean, and the second L-moment is half of the
    mean-absolute difference.

    Args:
        a: Array-like with samples of shape (n, ) or (m, n).
        k_max: The max order of the L-moments, >0.
        axis (optional): Axis along which to calculate the L-moments
        sort_kind (opional): The kind of sort algorithm to use.

    Returns:
        l: Array of shape (1 + k_max) or (m, 1 + k_max), where ``l[k]`` is
            the k-th L-moment (scalar or vector).

    """
    if k_max <= 0:
        raise ValueError('k_max must be a strictly positive integer')

    x: np.ndarray = np.sort(a, axis=axis, kind=sort_kind)

    if x.ndim > 1:
        assert axis is not None

        if x.ndim > 2:
            raise ValueError('sample array must be either 1-D or 2-D')

        if axis <= -1:
            axis += x.ndim

    n = x.shape[axis or 0]

    if k_max == 1:
        return np.array([1.0]) if x.ndim == 1 else np.ones(n, dtype=np.float_)

    if n < k_max:
        raise ValueError(
            f'the k-th L-moment requires at least k ({k_max}) samples, got {n}'
        )

    l_1k = (x if axis or x.ndim == 1 else x) @ l_weights(n, k_max)

    # prepend the 0th L-moments, i.e. 1
    return np.r_[1, l_1k] if l_1k.ndim == 1 else np.c_[np.ones(len(l_1k)), l_1k]


def l_moment(
    a: npt.ArrayLike,
    k: int,
    r: int = 0,
    /,
    axis: int | None = -1,
    *,
    sort_kind: SortKind | None = None,
) -> float | npt.NDArray[np.float_]:
    """
    Estimates the k-th L-moment, or the (k, r)-th L-moment ratio.

    Args:
        a: Array-like with samples of shape (n, ) or (m, n).
        k: The order of the L-moment.
        r (optional): If `r` is set to a positive integer, divides the k-th
            L-moment by the r-th one, and an L-moment ratio is returned.
            If r=0 (default), the regular L-moment is returned, as the 0th
            L-moment is always 1.
        axis (optional): Axis along which to calculate the L-moments
        sort_kind (opional): The kind of sort algorithm to use.

    Returns:
        l: A scalar or vector of size (m,); the k-th L-moment(s), optionally
            divided by the r-th L-moment(s).

    """

    ll = l_moments(a, max(k, r), axis=axis, sort_kind=sort_kind)

    i_kr = [k - 1, r - 1]
    lk, lr = ll[i_kr] if ll.ndim == 1 else ll[..., i_kr]

    if k == 0:
        return 1.0 / lr

    return lk if r == 0 or np.all(lk == 0) else lk / lr


def l_loc(
    a: npt.ArrayLike,
    /,
    axis: int | None = -1,
) -> float | npt.NDArray[np.float_]:
    """
    Sample L-location, i.e. the oh so familiar / boring arithmetic mean.
    """
    return l_moment(a, 1, axis=axis)


def l_scale(
    a: npt.ArrayLike,
    /,
    axis: int | None = -1,
    **kwargs,
) -> float | npt.NDArray[np.float_]:
    """
    Sample L-scale (analogue of standard deviation); the second L-moment, i.e.
    an alias of `moment(a, 2, ...)`.

    Equivalent to half the mean-absolute difference.
    """
    return l_moment(a, 2, axis=axis, **kwargs)


def l_skew(
    a: npt.ArrayLike,
    /,
    axis: int | None = -1,
    **kwargs,
) -> float | npt.NDArray[np.float_]:
    """
    L-skewness coefficient; the 3rd sample L-moment ratio.
    """
    return l_moment(a, 3, 2, axis=axis, **kwargs)


def l_kurt(
    a: npt.ArrayLike,
    /,
    axis: int | None = -1,
    **kwargs,
) -> float | npt.NDArray[np.float_]:
    """
    L-kurtosis coefficient; the 4th sample L-moment ratio.
    """
    return l_moment(a, 4, 2, axis=axis, **kwargs)
