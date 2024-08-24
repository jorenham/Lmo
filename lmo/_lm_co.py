from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np

import lmo.typing as lmt
from ._lm import l_weights
from ._utils import clean_order, clean_orders, ordered


if sys.version_info >= (3, 13):
    from typing import TypeVar, Unpack
else:
    from typing_extensions import TypeVar, Unpack

if TYPE_CHECKING:
    import optype.numpy as onpt

    import lmo.typing.np as lnpt


__all__ = (
    'l_comoment',
    'l_coratio',
    'l_costats',
    'l_coloc',
    'l_coscale',
    'l_corr',
    'l_coskew',
    'l_cokurtosis',
    'l_cokurt',
)


_T_scalar = TypeVar('_T_scalar', bound=np.generic)
_T_float = TypeVar('_T_float', bound=np.floating[Any], default=np.float64)
_DType: TypeAlias = np.dtype[_T_scalar] | type[_T_scalar]

_N0 = TypeVar('_N0', bound=int)
_N1 = TypeVar('_N1', bound=int)
_N2 = TypeVar('_N2', bound=int)
_Array2D: TypeAlias = np.ndarray[tuple[_N0, _N1], np.dtype[_T_scalar]]
_Array3D: TypeAlias = np.ndarray[tuple[_N0, _N1, _N2], np.dtype[_T_scalar]]


def l_comoment(
    a: lnpt.AnyVectorFloat | lnpt.AnyMatrixFloat,
    r: lmt.AnyOrder | lmt.AnyOrderND,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_T_float] = np.float64,
    rowvar: bool | None = None,
    sort: lnpt.SortKind | None = None,
    cache: bool | None = None,
) -> onpt.Array[Any, _T_float]:
    r"""
    Multivariate extension of [`lmo.l_moment`][lmo.l_moment].

    Estimates the L-comoment matrix:

    $$
    \Lambda_{r}^{(s, t)} =
        \left[
            \lambda_{r [ij]}^{(s, t)}
        \right]_{m \times m}
    $$

    Whereas the L-moments are calculated using the order statistics of the
    observations, i.e. by sorting, the L-comoment sorts $x_i$ using the
    order of $x_j$. This means that in general,
    $\lambda_{r [ij]}^{(s, t)} \neq \lambda_{r [ji]}^{(s, t)}$, i.e.
    $\Lambda_{r}^{(s, t)}$ is not symmetric.

    The $r$-th L-comoment $\lambda_{r [ij]}^{(s, t)}$ reduces to the
    L-moment if $i=j$, and can therefore be seen as a generalization of the
    (univariate) L-moments. Similar to how the diagonal of a covariance matrix
    contains the variances, the diagonal of the L-comoment matrix contains the
    L-moments.

    Based on the proposed definition by Serfling & Xiao (2007) for L-comoments.
    Extended to allow for generalized trimming.

    Parameters:
        a:
            1-D or 2-D array-like containing `m` variables and `n`
            observations.  Each row of `a` represents a variable, and each
            column a single observation of all those variables. Also see
            `rowvar` below.  If `a` is not an array, a conversion is attempted.
        r:
            The L-moment order(s), non-negative integer or array.
        trim:
            Left- and right-trim orders $(s, t)$, non-negative ints or
            floats that are bound by $s + t < n - r$.
        rowvar:
            If `True`, then each row (axis 0) represents a
            variable, with observations in the columns (axis 1).
            If `False`, the relationship is transposed: each column represents
            a variable, while the rows contain observations.
            If `None` (default), it is determined from the shape of `a`.
        dtype:
            Floating type to use in computing the L-moments. Default is
            [`numpy.float64`][numpy.float64].

        sort ('quick' | 'heap' | 'stable'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].
        cache:
            Set to `True` to speed up future L-moment calculations that have
            the same number of observations in `a`, equal `trim`, and equal or
            smaller `r`. By default, it will cache i.f.f. the trim is integral,
            and $r + s + t \le 24$. Set to `False` to always disable caching.

    Returns:
        L: Array of shape `(*r.shape, m, m)` with r-th L-comoments.

    Examples:
        Estimation of the second L-comoment (the L-coscale) from biviariate
        normal samples:

        >>> import lmo, numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.multivariate_normal([0, 0], [[6, -3], [-3, 3.5]], 99)
        >>> lmo.l_comoment(x, 2)
        array([[ 1.2766793 , -0.83299947],
               [-0.71547941,  1.05990727]])

        The diagonal contains the univariate L-moments:

        >>> lmo.l_moment(x, 2, axis=0)
        array([1.2766793 , 1.05990727])

        The orientation (`rowvar`) is automatically determined, unless
        explicitly specified.

        >>> lmo.l_comoment(x, 2).shape
        (2, 2)
        >>> lmo.l_comoment(x.T, 2).shape
        (2, 2)
        >>> lmo.l_comoment(x, 2, rowvar=False).shape
        (2, 2)
        >>> lmo.l_comoment(x.T, 2, rowvar=True).shape
        (2, 2)

    References:
        - [R. Serfling & P. Xiao (2007) - A Contribution to Multivariate
          L-Moments: L-Comoment Matrices](https://doi.org/10.1016/j.jmva.2007.01.008)
    """
    x = np.array(a, subok=True, ndmin=2)
    if x.ndim != 2:
        msg = f'sample array must be 2-D, got shape {x.shape}'
        raise ValueError(msg)

    if rowvar is None:
        rowvar = x.shape[0] < x.shape[1]
    if not rowvar:
        x = x.T
    m, n = x.shape

    if np.isscalar(r):
        _r = np.array(clean_order(cast(lmt.AnyOrder, r)))
    else:
        _r = clean_orders(cast(lmt.AnyOrderND, r))

    if not m:
        return np.empty((*np.shape(_r), 0, 0), dtype=dtype)

    r_min = int(np.min(_r))
    r_max = int(np.max(_r))

    if r_min == r_max == 0 and _r.ndim == 0:
        return np.identity(m, dtype=dtype)

    # projection/hat matrix of shape (r_max - r_min, n)
    p_k = l_weights(r_max, n, trim=trim, dtype=dtype, cache=cache)
    if r_min > 1:
        p_k = p_k[r_min - 1:]

    # L-comoment matrices for k = r_min, ..., r_max
    l_kij = np.empty((p_k.shape[0], m, m), dtype=dtype, order='F')

    for j in range(m):
        # *concomitants* of x[i] w.r.t. x[j] for all i
        x_kij = ordered(x, x[j], axis=-1, sort=sort or True)
        l_kij[:, :, j] = np.inner(p_k, x_kij)

    if r_min == 0:
        # the zeroth L-comoment is the delta function, so the L-comoment
        # matrix is the identity matrix
        l_0ij = np.identity(m, dtype=dtype)[None, :]
        return np.concat((l_0ij, l_kij)).take(_r, 0)

    return l_kij.take(_r - r_min, 0)


def l_coratio(
    a: lnpt.AnyVectorFloat | lnpt.AnyMatrixFloat,
    r: lmt.AnyOrder | lmt.AnyOrderND,
    s: lmt.AnyOrder | lmt.AnyOrderND,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_T_float] = np.float64,
    **kwds: Unpack[lmt.LComomentOptions],
) -> onpt.Array[Any, _T_float]:
    r"""
    Estimate the generalized matrix of L-comoment ratio's.

    $$
    \tilde \Lambda_{rk}^{(s, t)} =
        \left[
            \left. \lambda_{r [ij]}^{(s, t)} \right/
            \lambda_{k [ii]}^{(s, t)}
        \right]_{m \times m}
    $$

    See Also:
        - [`lmo.l_comoment`][lmo.l_comoment]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))
    l_r, l_s = l_comoment(a, rs, trim=trim, dtype=dtype, **kwds)
    return l_r / np.expand_dims(np.diagonal(l_s, axis1=-2, axis2=-1), -1)


def l_costats(
    a: lnpt.AnyVectorFloat | lnpt.AnyMatrixFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_T_float] = np.float64,
    **kwds: Unpack[lmt.LComomentOptions],
) -> _Array3D[Literal[4], Any, Any, _T_float]:
    """
    Calculates the L-*co*scale, L-corr(elation), L-*co*skew(ness) and
    L-*co*kurt(osis).

    Equivalent to `lmo.l_coratio(a, [2, 2, 3, 4], [0, 2, 2, 2], *, **)`.

    See Also:
        - [`lmo.l_stats`][lmo.l_stats]
        - [`lmo.l_coratio`][lmo.l_coratio]
    """
    r, s = [2, 2, 3, 4], [0, 2, 2, 2]
    return l_coratio(a, r, s, trim=trim, dtype=dtype, **kwds)


def l_coloc(
    a: lnpt.AnyVectorFloat | lnpt.AnyMatrixFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_T_float] = np.float64,
    **kwds: Unpack[lmt.LComomentOptions],
) -> _Array2D[Any, Any, _T_float]:
    r"""
    L-colocation matrix of 1st L-comoment estimates, $\Lambda^{(s, t)}_1$.

    Alias for [`lmo.l_comoment(a, 1, *, **)`][lmo.l_comoment].

    Notes:
        If `trim = (0, 0)` (default), the L-colocation for $[ij]$ is the
        L-location $\lambda_1$ of $x_i$, independent of $x_j$.

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

        What this tells us, is somewhat of a mystery: trimmed L-comoments have
        been only been briefly *mentioned* once or twice in the literature.


    See Also:
        - [`lmo.l_comoment`][lmo.l_comoment]
        - [`lmo.l_loc`][lmo.l_loc]
        - [`numpy.mean`][numpy.mean]
    """
    return l_comoment(a, 1, trim=trim, dtype=dtype, **kwds)


def l_coscale(
    a: lnpt.AnyVectorFloat | lnpt.AnyMatrixFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_T_float] = np.float64,
    **kwds: Unpack[lmt.LComomentOptions],
) -> _Array2D[Any, Any, _T_float]:
    r"""
    L-coscale matrix of 2nd L-comoment estimates, $\Lambda^{(s, t)}_2$.

    Alias for [`lmo.l_comoment(a, 2, *, **)`][lmo.l_comoment].

    Analogous to the (auto-) variance-covariance matrix, the L-coscale matrix
    is positive semi-definite, and its main diagonal contains the L-scale's.
    conversely, the L-coscale matrix is inherently asymmetric, thus yielding
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
    return l_comoment(a, 2, trim=trim, dtype=dtype, **kwds)


def l_corr(
    a: lnpt.AnyVectorFloat | lnpt.AnyMatrixFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_T_float] = np.float64,
    **kwds: Unpack[lmt.LComomentOptions],
) -> _Array2D[Any, Any, _T_float]:
    r"""
    Sample L-correlation coefficient matrix $\tilde\Lambda^{(s, t)}_2$;
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
    return l_coratio(a, 2, 2, trim=trim, dtype=dtype, **kwds)


def l_coskew(
    a: lnpt.AnyVectorFloat | lnpt.AnyMatrixFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_T_float] = np.float64,
    **kwds: Unpack[lmt.LComomentOptions],
) -> _Array2D[Any, Any, _T_float]:
    r"""
    Sample L-coskewness coefficient matrix $\tilde\Lambda^{(s, t)}_3$.

    Alias for [`lmo.l_coratio(a, 3, 2, *, **)`][lmo.l_coratio].

    See Also:
        - [`lmo.l_coratio`][lmo.l_coratio]
        - [`lmo.l_skew`][lmo.l_skew]
    """
    return l_coratio(a, 3, 2, trim=trim, dtype=dtype, **kwds)


def l_cokurtosis(
    a: lnpt.AnyVectorFloat | lnpt.AnyMatrixFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_T_float] = np.float64,
    **kwds: Unpack[lmt.LComomentOptions],
) -> _Array2D[Any, Any, _T_float]:
    r"""
    Sample L-cokurtosis coefficient matrix $\tilde\Lambda^{(s, t)}_4$.

    Alias for [`lmo.l_coratio(a, 4, 2, *, **)`][lmo.l_coratio].

    See Also:
        - [`lmo.l_coratio`][lmo.l_coratio]
        - [`lmo.l_kurtosis`][lmo.l_kurtosis]
    """
    return l_coratio(a, 4, 2, trim=trim, dtype=dtype, **kwds)


l_cokurt = l_cokurtosis
"""Alias for [`lmo.l_cokurtosis`][lmo.l_cokurtosis]."""
