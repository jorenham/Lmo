"""
The multivariate (co-) variants of the univariate L-moment estimators.

Based on the work by Robert Serﬂing and Peng Xiao - "A contribution to
multivariate L-moments: L-comoment matrices"
"""

__all__ = (
    'l_comoments',
    'l_comoment',
    'l_coscale',
    'l_corr',
    'l_coskew',
    'l_cokurt',
)

import numpy as np
from numpy import typing as npt

from ._l_stats import l_weights
from ._typing import SortKind


def l_comoments(
    a: npt.ArrayLike,
    k_max: int,
    /,
    *,
    rowvar: bool = True,
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

    w_1k = l_weights(n, k_max)  # (n, k)
    w_0k = np.c_[np.ones(n), w_1k]  # (n, 1 + k)

    ki2 = np.argsort(x, kind=sort_kind)

    # perhaps np.einsum is the way to go here...?
    # the contiguity is suboptimal here, as well
    l12 = np.empty((m, m, 1 + k_max), dtype=np.float_)
    for i2 in range(m):
        l12[:, i2] = x[:, ki2[i2]] @ w_0k

    return l12


def l_comoment(
    a: npt.ArrayLike,
    k: int,
    r: int = 0,
    /,
    **kwargs
) -> float | npt.NDArray[np.float_]:
    """
    Estimates the k-th L-comoment `L[i,j]` matrix, or the (k, r)-th L-comoment
    ratio matrix `L[i,j] / l[j]`.

    Args:
        a: A 2-D array containing `m` variables and `n` observations.
            Each row of `a` represents a variable, and each column a single
            observation of all those variables. Also see `rowvar` below.
        k: The amount of L-comoments with order `1, ..., k`  that is returned.
            Must be `>0`.
        r (optional): If `r` is set to a positive integer, divides the k-th
            L-comoment by the r-th one, and an L-cpmoment ratio is returned.
            If r=0 (default), the regular L-moment is returned, as the 0th
            L-comoment is always 1.

    Returns:
        L: Array of shape (m, m) with k-th L-comoment matrix, or the
            (k, r)-L-coratio matrix.

    """
    ln = l_comoments(a, max(k, r), **kwargs)
    return ln[k] / ln[r].diagonal()[:, np.newaxis] if r else ln[k]


def l_coscale(a: npt.ArrayLike, /, **kwargs) -> float | npt.NDArray[np.float_]:
    """
    L-coscale coefficients; the 2nd L-comoment matrix estimator.

    Analogous to the (auto-) covariance matrix, the L-coscale matrix is
    positive semi-definite, and its main diagonal contains the L-scale's.
    But unlike the variance-covariance matrix, the L-coscale matrix is not
    symmetric.
    """
    return l_comoment(a, 2, **kwargs)


def l_corr(a: npt.ArrayLike, /, **kwargs) -> float | npt.NDArray[np.float_]:
    """
    Sample L-correlation coefficient matrix; the ratio of the L-coscale matrix
    over the L-scale **column**-vectors, i.e. the L-correlation matrix is
    typically asymmetric.

    The coefficients are, similar to the pearson correlation, bounded
    within [-1, 1]. Where the pearson correlation coefficient measures
    linearity, the L-correlation coefficient measures monotonicity.
    """
    return l_comoment(a, 2, 2, **kwargs)


def l_coskew(a: npt.ArrayLike, /, **kwargs) -> float | npt.NDArray[np.float_]:
    """
    L-coskewness coefficients; the 3rd L-comoment ratio matrix estimator.

    The main diagonal cantains the L-skewness coefficients.
    """
    return l_comoment(a, 3, 2, **kwargs)


def l_cokurt(a: npt.ArrayLike, /, **kwargs) -> float | npt.NDArray[np.float_]:
    """
    L-cokurtosis coefficients; the 4th L-comoment ratio matrix estimator.

    The main diagonal contains the L-kurtosis coefficients.
    """
    return l_comoment(a, 4, 2, **kwargs)
