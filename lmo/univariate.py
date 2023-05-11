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

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from .weights import tl_weights
from ._typing import SortKind

_AnyFloat: TypeAlias = np.floating[Any]
_NumOrArr: TypeAlias = _AnyFloat | npt.NDArray[_AnyFloat]


def tl_moment(
    a: npt.ArrayLike,
    r: int,
    /,
    s: int = 1,
    t: int = 1,
    *,
    axis: int | None = None,
    sort: SortKind | None = None,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    The r-th sample TL-moment.

    Args:
        a: Array-like with samples.
        r: The order of the TL-moment; strictly positive integer.

        s: Amount of samples to trim at the start, default is 1.
        t: Amount of samples to trim at the end, default is 1.

        axis: Axis along wich to calculate the TL-moments.
            If `None` (default), all samples in the array will be used.
        sort: Sorting algorithm to use, default is `'quicksort'`. See
            `numpy.sort` for more info.

    Returns:
        Scalar or array; the r-th TL-moment(s).

    """
    x = np.sort(a, axis=axis, kind=sort)
    n = x.shape[axis or 0]

    if r == 0:
        # zeroth (TL-)moment is 1
        # noinspection PyTypeChecker
        dt: np.dtype[_AnyFloat] = np.find_common_type([x.dtype], [np.float_])
        return np.ones(x.size // n, dtype=dt) if x.ndim > 1 else dt.type(1)

    w = tl_weights(n, r, s, t)

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
    a: npt.ArrayLike,
    r: int,
    /,
    k: int = 2,
    s: int = 1,
    t: int = 1,
    *,
    axis: int | None = None,
    sort: SortKind | None = None,
) -> _NumOrArr:
    """
    Ratio of the r-th and k-th (2nd by default) sample TL-moments.
    """
    x = np.sort(a, axis=axis, kind=sort)

    l_r = tl_moment(x, r, s, t, axis=axis)

    if k == 0:
        return l_r
    if k == r:
        return np.ones_like(l_r)[()]

    l_k = l_r if k == r else tl_moment(x, k, s, t, axis=axis)

    # i.e. `x / 0 = 0 if x == 0 else np.nan`
    return np.divide(
        l_r,
        l_k,
        out=np.where(l_r == 0, 0.0, np.nan),
        where=l_k != 0
    )[()]  # [()] converts any 0-dimensional arrays to scalar


def tl_loc(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> _NumOrArr:
    """
    TL-location: the first sample TL-moment. Analogous to the sample mean.
    """
    return tl_moment(a, 1, s, t, **kwargs)


def tl_scale(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> _NumOrArr:
    """
    TL-scale: the second TL-moment. Analogous to the sample standard deviation.
    """
    return tl_moment(a, 2, s, t, **kwargs)


def tl_skew(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> _NumOrArr:
    """
    TL-skewness coefficient; the ratio of the 3rd and 2nd sample TL-moments.
    """
    return tl_ratio(a, 3, s=s, t=t, **kwargs)


def tl_kurt(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> _NumOrArr:
    """
    TL-kurtosis coefficient; the ratio of the 4th and 2nd sample TL-moments.
    """
    return tl_ratio(a, 4, s=s, t=t, **kwargs)


# L-moment aliasses

def l_moment(a: npt.ArrayLike, r: int, /, **kwargs: Any) -> _NumOrArr:
    """
    The r-th sample L-moment.
    Alias for ``tl_moment(a, r, 0, 0, **kwargs)``.
    """
    return tl_moment(a, r, 0, 0, **kwargs)


def l_ratio(
    a: npt.ArrayLike,
    r: int, k: int = 2,
    /,
    **kwargs: Any,
) -> _NumOrArr:
    """
    Ratio of the r-th and k-th L-moments.
    Alias for ``tl_ratio(a, r, k, 0, 0, **kwargs)``.

    For k > 0, the L-moment ratio's are bounded within [-1, 1].
    """
    return tl_ratio(a, r, k, 0, 0, **kwargs)


def l_loc(a: npt.ArrayLike, /, **kwargs: Any) -> _NumOrArr:
    """
    L-location: the first sample L-moment.
    Equivalent to the sample mean.
    """
    return l_moment(a, 1, **kwargs)


def l_scale(a: npt.ArrayLike, /, **kwargs: Any) -> _NumOrArr:
    """
    L-scale: the second L-moment.
    Equivalent to half the mean absolute difference.
    """
    return l_moment(a, 2, **kwargs)


def l_skew(a: npt.ArrayLike, /, **kwargs: Any) -> _NumOrArr:
    """
    L-skewness coefficient; the ratio of the 3rd and 2nd sample L-moments.
    """
    return l_ratio(a, 3, **kwargs)


def l_kurt(a: npt.ArrayLike, /, **kwargs: Any) -> _NumOrArr:
    """
    L-kurtosis coefficient; the ratio of the 4th and 2nd sample L-moments.
    """
    return l_ratio(a, 4, **kwargs)
