"""
The multivariate (co-) variants of the univariate L-moment estimators.

Based on the work by Robert Serﬂing and Peng Xiao - "A contribution to
multivariate L-moments: L-comoment matrices"
"""

__all__ = (
    'tl_comoment',
    'tl_coratio',
    'tl_coloc',
    'tl_coscale',
    'tl_corr',
    'tl_coskew',
    'tl_cokurt',

    'l_comoment',
    'l_coratio',
    'l_coloc',
    'l_coscale',
    'l_corr',
    'l_coskew',
    'l_cokurt',
)

from typing import Any, cast

import numpy as np
from numpy import typing as npt

from . import tl_moment
from ._typing import SortKind
from .weights import tl_weights


# noinspection PyPep8Naming
def tl_comoment(
    a: npt.ArrayLike,
    r: int,
    /,
    s: int = 1,
    t: int = 1,
    *,
    rowvar: bool = True,
    sort: SortKind | None = None,
) -> npt.NDArray[np.float_]:
    """
    Multivariate extension of the sample TL-moments: the TL-comoment matrix.

    Based on the proposed definition by Serfling & Xiao (2007) for L-moments.
    Modified to be compatible with (the more general) TL-moments.

    Args:
        a: A 1-D or 2-D array containing `m` variables and `n` observations.
          Each row of `a` represents a variable, and each column a single
          observation of all those variables. Also see `rowvar` below.
        r: The order of the TL-moment; strictly positive integer.

        s (optional): Amount of samples to trim at the start, default is 1.
        t (optional): Amount of samples to trim at the end, default is 1.

        rowvar (optional): If `rowvar` is True (default), then each row
          represents a variable, with observations in the columns. Otherwise,
          the relationship is transposed: each column represents a variable,
          while the rows contain observations.
        sort (opional): Sorting algorithm to use, default is 'quicksort'. See
          `numpy.sort` for more info.

    Returns:
        L: Matrix of shape (m, m) with r-th TL-comoments.

    References:
        * Serfling, R. and Xiao, P., 2006. A Contribution to Multivariate
          L-Moments: L-Comoment Matrices.

    """
    if r < 0:
        raise ValueError('r must be >=0')

    x = np.asanyarray(a)

    if x.ndim != 2:
        raise ValueError(f'sample array must be 2-D, got {x.ndim}')
    elif not rowvar:
        x = x.T

    m, n = x.shape
    dtype = np.find_common_type([x.dtype], [np.float_])

    if not m or not x.size:
        return np.empty((0, 0), dtype)

    if r == 0:
        # The zeroth (TL-)co-moment matrix is the identity matrix, right..?
        return np.eye(m, dtype=dtype)

    w = tl_weights(n, r, s, t)

    L_ji = np.empty((m, m), dtype=dtype)
    for j, ii in enumerate(np.argsort(x, kind=sort)):
        L_ji[j] = x[:, ii] @ w

    return L_ji.T


# noinspection PyPep8Naming
def tl_coratio(
    a: npt.ArrayLike,
    r: int,
    /,
    k: int = 2,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-comoment ratio matrix `L_r[i, j] / l_k[j]`.

    References:
        * Serfling, R. and Xiao, P., 2006. A Contribution to Multivariate
          L-Moments: L-Comoment Matrices.

    """
    L_k = tl_comoment(a, r, s, t, **kwargs)

    if k == 0:
        return L_k

    l_r = L_k.diagonal() if k == r else cast(
        npt.NDArray[np.float_], tl_moment(a, r, s, t, **kwargs)
    )

    return L_k / l_r[:, np.newaxis]


def tl_coloc(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-co-locations; the 1st TL-comoment matrix.

    Notes:
        * If you figure out how to interpret this, or how this can be applied,
          please tell me (github: @jorenham).

    """
    return tl_comoment(a, 1, s, t, **kwargs)


def tl_coscale(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-coscale coefficients; the 2nd TL-comoment matrix.

    Analogous to the (auto-) covariance matrix, the TL-coscale matrix is
    positive semi-definite, and its main diagonal contains the TL-scale's.
    But unlike the variance-covariance matrix, the TL-coscale matrix is not
    symmetric.
    """
    return tl_comoment(a, 2, s, t, **kwargs)


def tl_corr(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    Sample TL-correlation coefficient matrix; the ratio of the TL-coscale
    matrix over the TL-scale **column**-vectors, i.e. the TL-correlation matrix
    is typically asymmetric.

    The diagonal contains only ones.

    Notes:
        Where the pearson correlation coefficient measures linearity, the
        (T)L-correlation coefficient measures monotonicity.

    """
    return tl_coratio(a, 2, s=s, t=t, **kwargs)


def tl_coskew(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-coskewness coefficients; the 3rd TL-comoment ratio matrix.

    The main diagonal cantains the TL-skewness coefficients.
    """
    return tl_coratio(a, 3, s=s, t=t, **kwargs)


def tl_cokurt(
    a: npt.ArrayLike,
    /,
    s: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-cokurtosis coefficients; the 4th TL-comoment ratio matrix.

    The main diagonal contains the TL-kurtosis coefficients.
    """
    return tl_coratio(a, 4, s=s, t=t, **kwargs)


def l_comoment(
    a: npt.ArrayLike,
    r: int,
    /,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    The r-th sample L-comoment matrix.
    Alias for ``tl_comoment(a, r, 0, 0, **kwargs)``.
    """
    return tl_comoment(a, r, 0, 0, **kwargs)


def l_coratio(
    a: npt.ArrayLike,
    r: int,
    k: int = 2,
    /,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    L-comoment ratio matrix `L_r[i, j] / l_k[j]`.
    Alias for ``tl_coratio(a, r, k, 0, 0, **kwargs)``.
    """
    return tl_coratio(a, r, k, 0, 0, **kwargs)


def l_coloc(a: npt.ArrayLike, /, **kwargs: Any) -> npt.NDArray[np.float_]:
    """
    L-colocation matrix, i.e. each `L[i, j]` is the sample mean of `a[i]`.
    """
    return l_comoment(a, 1, **kwargs)


def l_coscale(a: npt.ArrayLike, /, **kwargs: Any) -> npt.NDArray[np.float_]:
    """
    L-scale: the second L-comoment matrix.
    """
    return l_comoment(a, 2, **kwargs)


def l_corr(a: npt.ArrayLike, /, **kwargs: Any) -> npt.NDArray[np.float_]:
    """
    L-correlation coefficients; the 2nd L-comoment ratio matrix.

    Unlike the TL-correlation, the L-correlation is bounded within [-1, 1].
    """
    return l_coratio(a, 2, **kwargs)


def l_coskew(a: npt.ArrayLike, /, **kwargs: Any) -> npt.NDArray[np.float_]:
    """
    L-coskewness coefficients; the 3rd L-comoment ratio matrix.
    """
    return l_coratio(a, 3, **kwargs)


def l_cokurt(a: npt.ArrayLike, /, **kwargs: Any) -> npt.NDArray[np.float_]:
    """
    L-cokurtosis coefficients; the 4th L-comoment ratio matrix.
    """
    return l_coratio(a, 4, **kwargs)
