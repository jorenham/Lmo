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

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from ._l_stats import l_weights
from ._typing import SortKind

_AnyFloat: TypeAlias = np.floating[Any]
_NumOrVec: TypeAlias = _AnyFloat | npt.NDArray[_AnyFloat]


def l_moments(
    a: npt.ArrayLike,
    k_max: int,
    *,
    sort_kind: SortKind | None = None,
) -> npt.NDArray[_AnyFloat]:
    """
    Estimate the [0, 1, ..., k_max]-th sample L-moments, where the 0th is 1,
    the first the arithmetic mean, and the second L-moment is half of the
    mean-absolute difference.

    Args:
        a: Array-like with samples of shape (n, ) or (n, m).
        k_max: The max order of the L-moments, >0.
        sort_kind (opional): The kind of sort algorithm to use.

    Returns:
        l: Array of shape (1 + k_max,) or (1 + k_max, [m]), where ``l[k]`` is
            the k-th L-moment (scalar or vector).

    """
    if k_max <= -1:
        raise ValueError('k_max must be a strictly positive integer')

    x: np.ndarray = np.sort(a, axis=0, kind=sort_kind)

    if x.ndim not in {1, 2}:
        raise ValueError('sample array must be either 1-D or 2-D')

    n = len(x)
    dtype = np.find_common_type([x.dtype], [np.float_])

    # the zeroth L-moment
    l0 = np.ones((1, ) if x.ndim == 1 else (1, x.shape[-1]), dtype=dtype)

    if k_max == 0:
        return l0

    if n < k_max:
        raise ValueError(
            f'the k-th L-moment requires at least k ({k_max}) samples, got {n}'
        )

    # the [1, ..., k_max]-th L-moments
    l1k = (l_weights(k_max, n) @ x).astype(dtype)
    # l1k = (x.T @ l_weights(k_max, n)).T.astype(dtype)

    # concat the [0]-th and the [1, ..., k_max]-th L-moments
    return np.r_[l0, l1k]


def l_moment(
    a: npt.ArrayLike,
    k: int,
    r: int = 0,
    /,
    **kwargs
) -> _NumOrVec:
    """
    Estimates the k-th L-moment, or the (k, r)-th L-moment ratio.

    Args:
        a: Array-like of shape (n, ) or (n, m) with `n` samples and `m`
            variables.
        k: The order of the L-moment.
        r (optional): If `r` is set to a positive integer, divides the k-th
            L-moment by the r-th one, and an L-moment ratio is returned.
            If r=0 (default), the regular L-moment is returned, as the 0th
            L-moment is always 1.

    Returns:
        l: A scalar or vector of size (m,); the k-th L-moment(s), optionally
            divided by the r-th L-moment(s).

    """
    if k <= -1 or r <= -1:
        raise ValueError('k and r must be strictly positive integers')

    ll = l_moments(a, max(k, r), **kwargs)
    assert np.all(ll[0] == 1)

    return ll[k] / ll[r] if r else ll[k]


def l_loc(a: npt.ArrayLike, /, **kwargs) -> _NumOrVec:
    """
    Sample L-location, i.e. the oh so familiar / boring arithmetic mean.
    """
    return l_moment(a, 1, **kwargs)


def l_scale(a: npt.ArrayLike, /, **kwargs) -> _NumOrVec:
    """
    Sample L-scale (analogue of standard deviation); the second L-moment, i.e.
    an alias of `moment(a, 2, ...)`.

    Equivalent to half the mean-absolute difference.
    """
    return l_moment(a, 2, **kwargs)


def l_skew(a: npt.ArrayLike, /, **kwargs) -> _NumOrVec:
    """
    L-skewness coefficient; the 3rd sample L-moment ratio.
    """
    return l_moment(a, 3, 2, **kwargs)


def l_kurt(a: npt.ArrayLike, /, **kwargs) -> _NumOrVec:
    """
    L-kurtosis coefficient; the 4th sample L-moment ratio.
    """
    return l_moment(a, 4, 2, **kwargs)
