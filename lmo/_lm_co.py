__all__ = (
    'l_comoment',
    'l_coratio',
    'l_coloc',
    'l_coscale',
    'l_corr',
    'l_coskew',
    'l_cokurtosis',
)

from typing import Any, TypeVar, cast

import numpy as np
from numpy import typing as npt

from ._utils import clean_order
from ._lm import l_weights
from .stats import order_stats
from .typing import AnyInt, IntVector, SortKind

T = TypeVar('T', bound=np.floating[Any])


def l_comoment(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    /,
    trim: tuple[int, int] = (0, 0),
    rowvar: bool = True,
    dtype: np.dtype[T] | type[T] = np.float_,
    *,
    sort: SortKind | None = 'stable',
) -> npt.NDArray[T]:
    """
    Multivariate extension of [`lmo.l_moment`][lmo.l_moment]. Estimates the
    L-comoment matrix:

    $$
    \\Lambda_{r}^{(t_1, t_2)} =
        \\left[
            \\lambda_{r [ij]}^{(t_1, t_2)}
        \\right]_{m \\times m}
    $$

    Whereas the L-moments are calculated using the order statistics of the
    observations, i.e. by sorting, the L-comoment sorts $x_i$ using the
    order of $x_j$. This means that in general,
    $\\lambda_{r [ij]}^{(t_1, t_2)} \\neq \\lambda_{r [ji]}^{(t_1, t_2)}$, i.e.
    $\\Lambda_{r}^{(t_1, t_2)}$ is not symmetric.

    The $r$-th L-comoment $\\lambda_{r [ij]}^{(t_1, t_2)}$ reduces to the
    L-moment if $i=j$, and can therefore be seen as a generalization of the
    (univariate) L-moments. Similar to how the diagonal of a covariance matrix
    contains the variances, the diagonal of the L-comoment matrix contains the
    L-moments.

    Based on the proposed definition by Serfling & Xiao (2007) for L-comoments.
    Extended to allow for generalized trimming.

    Parameters:
        a:
            1-D or 2-D array-like containing `m` variables and `n` observations.
            Each row of `a` represents a variable, and each column a single
            observation of all those variables. Also see `rowvar` below.
            If `a` is not an array, a conversion is attempted.

        r:
            The L-moment order(s), non-negative integer or array.

        trim:
            Left- and right-trim orders $(t_1, t_2)$, non-negative integers
            that are bound by $t_1 + t_2 < n - r$.

            Some special cases include:

            - $(0, 0)$: The original **L**-moment, introduced by Hosking (1990).
                Useful for fitting the e.g. log-normal and generalized extreme
                value (GEV) distributions.
            - $(0, m)$: **LL**-moment (**L**inear combination of **L**owest
                order statistics), instroduced by Bayazit & Onoz (2002).
                Assigns more weight to smaller observations.
            - $(s, 0)$: **LH**-moment (**L**inear combination of **H**igher
                order statistics), by Wang (1997).
                Assigns more weight to larger observations.
            - $(t, t)$: **TL**-moment (**T**rimmed L-moment) $\\lambda_r^t$,
                with symmetric trimming. First introduced by
                Elamir & Seheult (2003).
                Generally more robust than L-moments.
                Useful for fitting heavy-tailed distributions, such as the
                Cauchy distribution.

        rowvar:
            If `rowvar` is True (default), then each row (axis 0) represents a
            variable, with observations in the columns (axis 1).
            Otherwise, the relationship is transposed: each column represents
            a variable, while the rows contain observations.

        dtype:
            Floating type to use in computing the L-moments. Default is
            [`numpy.float64`][numpy.float64].

        sort ('quick' | 'stable' | 'heap'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].

    Returns:
        L: Array of shape `(*r.shape, m, m)` with r-th L-comoments.

    Examples:
        Estimation of the second L-comoment (the L-coscale) from biviariate
        normal samples:

        >>> import lmo, numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.multivariate_normal([0, 0], [[6, -3], [-3, 3.5]], 99).T
        >>> lmo.l_comoment(x, 2)
        array([[ 1.2766793 , -0.83299947],
               [-0.71547941,  1.05990727]])

        The diagonal contains the univariate L-moments:

        >>> lmo.l_moment(x, 2, axis=-1)
        array([1.2766793 , 1.05990727])

    References:
        * [R. Serfling & P. Xiao (2007) - A Contribution to Multivariate
            L-Moments: L-Comoment Matrices](https://doi.org/10.1016/j.jmva.2007.01.008)

    """
    def _clean_array(arr: npt.ArrayLike) -> npt.NDArray[T]:
        out = np.asanyarray(arr, dtype=dtype)
        return out if rowvar else out.T

    x = np.atleast_2d(_clean_array(a))
    if x.ndim != 2:
        raise ValueError(f'sample array must be 2-D, got {x.ndim}')

    _r = np.asarray(r)
    r_max: int = clean_order(cast(
        int,
        np.max(_r)  # pyright: ignore [reportUnknownMemberType]
    ))
    m, n = x.shape

    if not m:
        return np.empty(np.shape(_r) + (0, 0), dtype=dtype)

    # projection matrix of shape (r, n)
    P_r = l_weights(r_max, n, trim, dtype=dtype)

    # L-comoment matrices for r = 0, ..., r_max
    L_ij = np.empty((r_max + 1, m, m), dtype=dtype)

    # the zeroth L-comoment is the delta function, so the L-comoment
    # matrix is the identity matrix
    L_ij[0] = np.eye(m, dtype=dtype)

    kwargs = {'axis': -1, 'dtype': dtype, 'sort': sort}
    for j in range(m):
        # concomitants of x[i] w.r.t. x[j] for all i
        x_k_ij = order_stats(x, x[j], **kwargs)

        L_ij[1:, :, j] = np.inner(P_r, x_k_ij)

    return L_ij.take(_r, 0)  # pyright: ignore [reportUnknownMemberType]


def l_coratio(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: tuple[int, int] = (0, 0),
    rowvar: bool = True,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    Estimate the generalized matrix of L-comoment ratio's:

    $$
    \\tilde \\Lambda_{rs}^{(t_1, t_2)} =
        \\left[
            \\left. \\lambda_{r [ij]}^{(t_1, t_2)} \\right/
            \\lambda_{s [jj]}^{(t_1, t_2)}
        \\right]_{m \\times m}
    $$

    See Also:
        - [`lmo.l_comoment`][lmo.l_comoment]
        - [`lmo.l_ratio`][lmo.l_ratio]

    """
    rs = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))
    L_r, L_s = l_comoment(a, rs, trim, rowvar=rowvar, dtype=dtype, **kwargs)

    l_s = np.diagonal(L_s, axis1=-2, axis2=-1)

    return L_r / np.expand_dims(l_s, -1)


def l_coloc(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    rowvar: bool = True,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    L-colocation matrix of 1st L-comoment estimates, $\\Lambda^{(t_1, t_2)}_1$.

    Alias for [`lmo.l_comoment(a, 1, *, **)`][lmo.l_comoment].

    Notes:
        If `trim = (0, 0)` (default), the L-colocation for $[ij]$ is the
        L-location $\\lambda_1$ of $x_i$, independent of $x_j$.

    Examples:
        Without trimming, the L-colocation only provides marginal information:

        >>> import lmo, numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.multivariate_normal([0, 0], [[6, -3], [-3, 3.5]], 99).T
        >>> lmo.l_loc(x, axis=-1)
        array([-0.02678225,  0.03008309])
        >>> lmo.l_coloc(x)
        array([[-0.02678225, -0.02678225],
               [ 0.03008309,  0.03008309]])

        But the trimmed L-locations are a different story...

        >>> lmo.l_loc(x, trim=(1, 1), axis=-1)
        array([-0.10488868, -0.00625729])
        >>> lmo.l_coloc(x, trim=(1, 1))
        array([[-0.10488868, -0.03797989],
               [ 0.03325074, -0.00625729]])

        What this tells us, is somehwat of a mystery: trimmed L-comoments have
        been only been briefly *mentioned* once or twice in the literature.


    See Also:
        - [`lmo.l_comoment`][lmo.l_comoment]
        - [`lmo.l_loc`][lmo.l_loc]
        - [`numpy.mean`][numpy.mean]

    """
    return l_comoment(a, 1, trim, rowvar=rowvar, dtype=dtype, **kwargs)


def l_coscale(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    rowvar: bool = True,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    L-coscale matrix of 2nd L-comoment estimates, $\\Lambda^{(t_1, t_2)}_2$.

    Alias for [`lmo.l_comoment(a, 2, *, **)`][lmo.l_comoment].

    Analogous to the (auto-) variance-covariance matrix, the L-coscale matrix
    is positive semi-definite, and its main diagonal contains the L-scale's.
    conversely, the L-coscale matrix is inherently assymmetric, thus yielding
    more information.

    Examples:
        >>> import lmo, numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.multivariate_normal([0, 0], [[6, -3], [-3, 3.5]], 99).T
        >>> lmo.l_scale(x, trim=(1, 1), axis=-1)
        array([0.66698774, 0.54440895])
        >>> lmo.l_coscale(x, trim=(1, 1))
        array([[ 0.66698774, -0.41025416],
               [-0.37918065,  0.54440895]])

    See Also:
        - [`lmo.l_comoment`][lmo.l_comoment]
        - [`lmo.l_scale`][lmo.l_scale]
        - [`numpy.cov`][numpy.cov]

    """
    return l_comoment(a, 2, trim, rowvar=rowvar, dtype=dtype, **kwargs)


def l_corr(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    rowvar: bool = True,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    Sample L-correlation coefficient matrix $\\tilde\\Lambda^{(t_1, t_2)}_2$;
    the ratio of the L-coscale matrix over the L-scale **column**-vectors.

    Alias for [`lmo.l_coratio(a, 2, 2, *, **)`][lmo.l_coratio].

    The diagonal consists of all 1's.

    Where the pearson correlation coefficient measures linearity, the
    (T)L-correlation coefficient measures monotonicity.

    Examples:
        >>> import lmo, numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> cov = np.array([[6, -3], [-3, 3.5]])
        >>> x = rng.multivariate_normal([0, 0], [[6, -3], [-3, 3.5]], 99).T
        >>> lmo.l_corr(x)
        array([[ 1.        , -0.65247355],
               [-0.67503962,  1.        ]])

        Let's compare this with the theoretical correlation

        >>> cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        -0.6546536707079772

        and the (Pearson) correlation coefficient matrix:

        >>> np.corrcoef(x)
        array([[ 1.        , -0.66383285],
               [-0.66383285,  1.        ]])



    See Also:
        - [`lmo.l_coratio`][lmo.l_coratio]
        - [`numpy.corrcoef`][numpy.corrcoef]

    """
    return l_coratio(a, 2, 2, trim, rowvar=rowvar, dtype=dtype, **kwargs)


def l_coskew(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    rowvar: bool = True,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    Sample L-coskewness coefficient matrix $\\tilde\\Lambda^{(t_1, t_2)}_3$.

    Alias for [`lmo.l_coratio(a, 3, 2, *, **)`][lmo.l_coratio].

    See Also:
        - [`lmo.l_coratio`][lmo.l_coratio]
        - [`lmo.l_skew`][lmo.l_skew]

    """
    return l_coratio(a, 3, 2, trim, rowvar=rowvar, dtype=dtype, **kwargs)


def l_cokurtosis(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    rowvar: bool = True,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    Sample L-cokurtosis coefficient matrix $\\tilde\\Lambda^{(t_1, t_2)}_4$.

    Alias for [`lmo.l_coratio(a, 4, 2, *, **)`][lmo.l_coratio].

    See Also:
        - [`lmo.l_coratio`][lmo.l_coratio]
        - [`lmo.l_kurtosis`][lmo.l_kurtosis]

    """
    return l_coratio(a, 4, 2, trim, rowvar=rowvar, dtype=dtype, **kwargs)
