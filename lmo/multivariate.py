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

from . import univariate, weights as _weights
from .typing import AnyMatrix, SortKind, Trimming


# noinspection PyPep8Naming
def tl_comoment(
    a: AnyMatrix,
    r: int,
    /,
    trim: Trimming = 1,
    rowvar: bool = True,
    *,
    weights: AnyMatrix | None = None,
    sort: SortKind | None = None,
) -> npt.NDArray[np.float_]:
    """
    Multivariate extension of the sample TL-moments. Calculates the
    TL-comoment matrix of sample TL-comoments:

    $$
    \\mathbf \\Lambda_{r}^{(t_1, t_2)} =
        \\bigg[
            \\lambda_{r [ij]}^{(t_1, t_2)}
        \\bigg]_{m \\times m}
    $$

    Whereas the TL-moments are calculated using the order statistics of the
    observations, i.e. by sorting, the TL-comoment sorts $x_i$ using the
    order of $x_j$. This means that in general,
    $\\lambda_{r [ij]}^{(t_1, t_2)} \\neq \\lambda_{r [ji]}^{(t_1, t_2)}$, i.e.
    $\\mathbf \\Lambda_{r}^{(t_1, t_2)}$ is not symmetric.

    The $r$-th TL-comoment $\\lambda_{r [ij]}^{(t_1, t_2)}$ reduces to the
    TL-moment if $i=j$, and can therefore be seen as a generalization of the
    (univariate) TL-moments. Similar to how the diagonal of a covariance matrix
    contains the variances, the diagonal of the TL-comoment matrix contains the
    TL-moments.

    Based on the proposed definition by Serfling & Xiao (2007) for L-comoments.
    Modified to be compatible with (the more general) TL-moments.

    Parameters:
        a (array_like):
            A 2-D array-like containing `m` variables and `n` observations.
            Each row of `a` represents a variable, and each column a single
            observation of all those variables. Also see `rowvar` below.

        r:
            The order of the TL-moment, a strictly positive integer.

        trim (int | tuple[int, int]):
            Amount of samples to trim as either

            - `t: int` for symmetric trimming, equivalent to `(t, t)`.
            - `(t1: int, t2: int)` for asymmetric trimming, or

            If `0` is passed, the L-comoment is returned.

        rowvar:
            If `rowvar` is True (default), then each row
            represents a variable, with observations in the columns. Otherwise,
            the relationship is transposed: each column represents a variable,
            while the rows contain observations.

        weights (array_like, optional):
            An 1-D or 2-D array-like of weights associated with the values in
            `a`.
            Each value in `a` contributes to the average according to its
            associated weight.
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
        L: Array of shape `(m, m)` with r-th TL-comoments.

    References:
        * Serfling, R. and Xiao, P., 2006. A Contribution to Multivariate
          L-Moments: L-Comoment Matrices.

    """
    if r < 0:
        raise ValueError('r must be >=0')

    # ensure `x` is shape (v, n) and `w_x` is None or of shape `(v, n)`
    x = np.asanyarray(a)
    if weights is None:
        w_x = None
    else:
        x, w_x = np.broadcast_arrays(x, np.asanyarray(weights))

    if x.ndim != 2:
        raise ValueError(f'sample array must be 2-D, got {x.ndim}')
    elif not rowvar:
        x = x.T
        if w_x is not None:
            w_x = w_x.T

    m, n = x.shape
    dtype = np.result_type(x, np.float_)

    if not m or not x.size:
        return np.empty((0, 0), dtype)

    if r == 0:
        # The zeroth (TL-)co-moment matrix is the identity matrix, right..?
        return np.eye(m, dtype=dtype)

    # vector of size n
    w_r = _weights.tl_weights(n, r, trim, dtype=dtype)

    # x[i] is the "concotaminant" w.r.t. x[j], i.e. x[i] is sorted using the
    # ordering from x[j]
    L_ij = np.empty((m, m), dtype=dtype, order='F')
    for j, ii in enumerate(np.argsort(x, kind=sort)):
        w = w_r if w_x is None else _weights.reweight(w_r, w_x[j, ii])
        L_ij[:, j] = x[:, ii] @ w

    return L_ij


# noinspection PyPep8Naming
def tl_coratio(
    a: AnyMatrix,
    r: int,
    /,
    k: int = 2,
    trim: Trimming = 1,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-comoment ratio matrix `L_r[i, j] / l_k[j]`.

    References:
        - Serfling, R. and Xiao, P., 2007. A Contribution to Multivariate
            L-Moments: L-Comoment Matrices.

    """
    L_k = tl_comoment(a, r, trim, rowvar=rowvar, **kwargs)

    if k == 0:
        return L_k

    axis = +rowvar
    l_r = cast(npt.NDArray[np.float_], (
        np.diag(L_k) if k == r
        else univariate.tl_moment(a, k, trim, axis=axis, **kwargs)
    ))

    return L_k / l_r[:, np.newaxis]


def tl_coloc(
    a: AnyMatrix,
    /,
    trim: Trimming = 1,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-co-locations; the 1st TL-comoment matrix.

    Notes:
        - If you figure out how to interpret this, or how this can be applied,
            please tell me (github: @jorenham).

    """
    return tl_comoment(a, 1, trim, rowvar=rowvar, **kwargs)


def tl_coscale(
    a: AnyMatrix,
    /,
    trim: Trimming = 1,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-coscale coefficients; the 2nd TL-comoment matrix.

    Analogous to the (auto-) covariance matrix, the TL-coscale matrix is
    positive semi-definite, and its main diagonal contains the TL-scale's.
    But unlike the variance-covariance matrix, the TL-coscale matrix is not
    symmetric.
    """
    return tl_comoment(a, 2, trim, rowvar=rowvar, **kwargs)


def tl_corr(
    a: AnyMatrix,
    /,
    trim: Trimming = 1,
    rowvar: bool = True,
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
    return tl_coratio(a, 2, trim=trim, rowvar=rowvar, **kwargs)


def tl_coskew(
    a: AnyMatrix,
    /,
    trim: Trimming = 1,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-coskewness coefficients; the 3rd TL-comoment ratio matrix.

    The main diagonal cantains the TL-skewness coefficients.
    """
    return tl_coratio(a, 3, trim=trim, rowvar=rowvar, **kwargs)


def tl_cokurt(
    a: AnyMatrix,
    /,
    trim: Trimming = 1,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    TL-cokurtosis coefficients; the 4th TL-comoment ratio matrix.

    The main diagonal contains the TL-kurtosis coefficients.
    """
    return tl_coratio(a, 4, trim=trim, rowvar=rowvar, **kwargs)


def l_comoment(
    a: AnyMatrix,
    r: int,
    /,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    The r-th sample L-comoment matrix.
    Alias for ``tl_comoment(a, r, 0, 0, **kwargs)``.
    """
    return tl_comoment(a, r, 0, rowvar=rowvar, **kwargs)


def l_coratio(
    a: AnyMatrix,
    r: int,
    k: int = 2,
    /,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    L-comoment ratio matrix `L_r[i, j] / l_k[j]`.
    Alias for ``tl_coratio(a, r, k, 0, 0, **kwargs)``.
    """
    return tl_coratio(a, r, k, 0, rowvar=rowvar, **kwargs)


def l_coloc(
    a: AnyMatrix,
    /,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    L-colocation matrix, i.e. each `L[i, j]` is the sample mean of `a[i]`.
    """
    return l_comoment(a, 1, rowvar=rowvar, **kwargs)


def l_coscale(
    a: AnyMatrix,
    /,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    L-scale: the second L-comoment matrix.
    """
    return l_comoment(a, 2, rowvar=rowvar, **kwargs)


def l_corr(
    a: AnyMatrix,
    /,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    L-correlation coefficients; the 2nd L-comoment ratio matrix.

    Unlike the TL-correlation, the L-correlation is bounded within [-1, 1].
    """
    return l_coratio(a, 2, rowvar=rowvar, **kwargs)


def l_coskew(
    a: AnyMatrix,
    /,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    L-coskewness coefficients; the 3rd L-comoment ratio matrix.
    """
    return l_coratio(a, 3, rowvar=rowvar, **kwargs)


def l_cokurt(
    a: AnyMatrix,
    /,
    rowvar: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    L-cokurtosis coefficients; the 4th L-comoment ratio matrix.
    """
    return l_coratio(a, 4, rowvar=rowvar, **kwargs)
