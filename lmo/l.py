"""
Estimators of the sample L-moments, and related summary statistics.

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

__all__ = (
    'moments',
    'moment',
    'scale',
    'skew',
    'kurt',

    'comoments',
    'comoment',
    'coscale',
    'corr',
    'coskew',
    'cokurt',

)

from typing import TypeAlias, Literal

import numpy as np
import numpy.typing as npt

from ._meta import jit
from .special import sh_legendre


SortKind: TypeAlias = Literal['quicksort', 'heapsort', 'stable']


@jit
def _l_weights(n: int, k_max: int, /) -> npt.NDArray[np.float_]:
    """
    L-moment linear sample weights.

    Equivalent to (but numerically more unstable than) this "naive" version::

        w = np.zeros((n, k))
        for r in range(1, n+1):
            for l in range(1, k+1):
                for j in range(min(r, l)):
                    w[r-1, l-1] += (
                        (-1) ** (l-1-j)
                        * comb(l-1, j) * comb(l-1+j, j)
                        * comb(r-1, j) / comb(n-1, j)
                    )
        return w / n

    Args:
        n: Amount of observations.
        k_max: Max degree of the L-moment, s.t. `0 <= k < n`.

    Returns:
        w: Weights of shape (n, k)

    """
    assert 0 < k_max < n

    w_kn = np.zeros((n, k_max), np.float_)
    w_kn[:, 0] = 1 / n
    for j in range(k_max - 1):
        w_kn[j:, j+1] = w_kn[j:, j] * np.arange(0, n-j, 1, np.float_) / (n-j)

    return w_kn @ sh_legendre(k_max).astype(np.float_).T


def moments(
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

    l_1k = (x if axis or x.ndim == 1 else x) @ _l_weights(n, k_max)

    # prepend the 0th L-moments, i.e. 1
    return np.r_[1, l_1k] if l_1k.ndim == 1 else np.c_[np.ones(len(l_1k)), l_1k]


def moment(
    a: npt.ArrayLike,
    k: int,
    /,
    r: int = 0,
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

    ll = moments(a, max(k, r), axis=axis, sort_kind=sort_kind)

    i_kr = [k - 1, r - 1]
    lk, lr = ll[i_kr] if ll.ndim == 1 else ll[..., i_kr]

    if k == 0:
        return 1.0 / lr

    return lk if r == 0 or np.all(lk == 0) else lk / lr


def scale(
    a: npt.ArrayLike,
    /,
    axis: int | None = -1,
    sort_kind: SortKind | None = None,
) -> float | npt.NDArray[np.float_]:
    """
    Sample L-scale (analogue of standard deviation); the second L-moment, i.e.
    an alias of `moment(a, 2, ...)`.

    Equivalent to half the mean-absolute difference.
    """
    return moment(a, 2, axis=axis, sort_kind=sort_kind)


def skew(
    a: npt.ArrayLike,
    /,
    axis: int | None = -1,
    sort_kind: SortKind | None = None,
) -> float | npt.NDArray[np.float_]:
    """
    Sample L-skewness.
    """
    return moment(a, 3, 2, axis=axis, sort_kind=sort_kind)


def kurt(
    a: npt.ArrayLike,
    /,
    axis: int | None = -1,
    sort_kind: SortKind | None = None,
) -> float | npt.NDArray[np.float_]:
    """
    Sample L-kurtosis.
    """
    return moment(a, 4, 2, axis=axis, sort_kind=sort_kind)


def comoments(
    a: npt.ArrayLike,
    k_max: int,
    /,
    rowvar: bool = True,
    *,
    sort_kind: SortKind | None = None,
) -> npt.NDArray[np.float_]:
    """
    Multivariate L-moments, the L-comoments, following the definition proposed
    by Serfling & Xiao (2007).

    Args:
        a: A 1-D or 2-D array containing `m` variables and `n` observations.
            Each row of `a` represents a variable, and each column a single
            observation of all those variables. Also see `rowvar` below.
        k_max: The max L-comoments order `0, 1, ..., k_max`  that is returned.
            Must be >0.
        rowvar (optional): If `rowvar` is True (default), then each row
            represents a variable, with observations in the columns. Otherwise,
            the relationship is transposed: each column represents a variable,
            while the rows contain observations.
        sort_kind (optional): Sorting algorithm. The default is 'quicksort'.
            Note that both 'stable' and 'mergesort' use timsort under the
            covers and, in general, the actual implementation will vary with
            data type.

    Returns:
        l12: Array of shape (m, m, 1 + k_max) with k-th L-comoments.

    See Also:
        * https://sciencedirect.com/science/article/pii/S0047259X07000103

    """
    if k_max < 1:
        raise ValueError('k must be >=1')

    x = np.asanyarray(a, np.float_)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    elif x.ndim != 2:
        raise ValueError('input must be 1-D or 2-D array-like')
    elif not rowvar:
        x = x.T

    m, n = x.shape
    if n <= k_max:
        raise ValueError(f'> k samples required, got {n}<={k_max}')

    w_1k = _l_weights(n, k_max)  # (n, k)
    w_0k = np.c_[np.ones(n), w_1k]  # (n, 1 + k)

    ki2 = np.argsort(x, kind=sort_kind)

    # perhaps np.einsum is the way to go here...?
    # the contiguity is suboptimal here, as well
    l12 = np.empty((m, m, 1 + k_max), dtype=np.float_)
    for i2 in range(m):
        l12[:, i2] = x[:, ki2[i2]] @ w_0k

    return l12


def comoment(
    a: npt.ArrayLike,
    k: int,
    /,
    r: int = 0,
    *,
    rowvar: bool = True,
    sort_kind: SortKind | None = None,
) -> float | npt.NDArray[np.float_]:
    """
    Estimates the k-th L-comoment, or the (k, r)-th L-comoment ratio.

    Args:
        a: A 1-D or 2-D array containing `m` variables and `n` observations.
            Each row of `a` represents a variable, and each column a single
            observation of all those variables. Also see `rowvar` below.
        k: The amount of L-comoments with order `1, ..., k`  that is returned.
            Must be `>=1`.
        r (optional): If `r` is set to a positive integer, divides the k-th
            L-comoment by the r-th one, and an L-cpmoment ratio is returned.
            If r=0 (default), the regular L-moment is returned, as the 0th
            L-comoment is always 1.
        rowvar (optional): If `rowvar` is True (default), then each row
            represents a variable, with observations in the columns. Otherwise,
            the relationship is transposed: each column represents a variable,
            while the rows contain observations.
        sort_kind (opional): The kind of sort algorithm to use.

    Returns:
        l12: Array of shape (m, m) with k-th L-comoment matrix, or the
            (k, r)-L-coratio matrix.

    """
    ln = comoments(a, max(k, r), rowvar=rowvar, sort_kind=sort_kind)

    # l_k[1, 2] / l_k[1]
    return ln[k] / ln[r].diagonal()[:, np.newaxis] if r else ln[k]


def coscale(a: npt.ArrayLike, /, **kwargs) -> float | npt.NDArray[np.float_]:
    """Sample L-coscale (covariance analogue) matrix."""
    return comoment(a, 2, **kwargs)


def corr(a: npt.ArrayLike, /, **kwargs) -> float | npt.NDArray[np.float_]:
    """
    Sample L-correlation coefficient matrix; the ratio of the L-coscale matrix
    over the L-scale **column**-vectors, i.e. the L-correlation matrix is
    typically asymmetric.

    The coefficients are, similar to the pearson correlation, bounded
    within [-1, 1]. Where the pearson correlation coefficient measures
    linearity, the L-correlation coefficient measures monotonicity.
    """
    return comoment(a, 2, 2, **kwargs)


def coskew(a: npt.ArrayLike, /, **kwargs) -> float | npt.NDArray[np.float_]:
    """
    L-coskewness coefficient matrix.
    """
    return comoment(a, 3, 2, **kwargs)


def cokurt(a: npt.ArrayLike, /, **kwargs) -> float | npt.NDArray[np.float_]:
    """
    L-cokurtosis coefficient matrix.
    """
    return comoment(a, 4, 2, **kwargs)
