"""
Unbiased sample estimators of the generalized trimmed L-moments.
"""

__all__ = (
    'l_moment',
    'l_ratio',
    'l_loc',
    'l_scale',
    'l_variation',
    'l_skew',
    'l_kurtosis',
)

from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt

from ._utils import clean_order, ensure_axis_at
from .stats import order_stats
from .typing import AnyInt, IntVector, SortKind
from .weights import l_weights

T = TypeVar('T', bound=np.floating[Any])


def l_moment(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    *,
    fweights: IntVector | None = None,
    aweights: npt.ArrayLike | None = None,
    sort: SortKind | None = 'stable',
) -> T | npt.NDArray[T]:
    """
    Estimates the generalized trimmed L-moment $\\lambda^{(t_1, t_2)}_r$ from
    the samples along the specified axis. By default, this will be the regular
    L-moment, $\\lambda_r = \\lambda^{(0, 0)}_r$.

    Parameters:
        a:
            Array containing numbers whose L-moments is desired.
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

        axis:
            Axis along wich to calculate the moments. If `None` (default),
            all samples in the array will be used.

        dtype:
            Floating type to use in computing the L-moments. Default is
            [`numpy.float64`][numpy.float64].

        fweights:
            1-D array of integer frequency weights; the number of times each
            observation vector should be repeated.

        aweights:
            An array of weights associated with the values in `a`. Each value
            in `a` contributes to the average according to its associated
            weight.
            The weights array can either be 1-D (in which case its length must
            be the size of a along the given axis) or of the same shape as `a`.
            If `aweights=None` (default), then all data in `a` are assumed to
            have a weight equal to one.

            All `aweights` must be `>=0`, and the sum must be nonzero.

            The algorithm is similar to that for weighted quantiles.

        sort ('quick' | 'stable' | 'heap'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].

    Returns:
        l:
            The L-moment(s) of the input This is a scalar iff a is 1-d and
            r is a scalar. Otherwise, this is an array with
            `np.ndim(r) + np.ndim(a) - 1` dimensions and shape like
            `(*np.shape(r), *(d for d in np.shape(a) if d != axis))`.

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_normal(20)
        >>> lmo.l_moment(x, [1, 2, 3, 4])
        array([0.00106117, 0.65354263, 0.01436636, 0.04280225])
        >>> lmo.l_moment(x, [1, 2, 3, 4], trim=(1, 1))
        array([-0.0133052 ,  0.36644423, -0.00823471, -0.01034343])

    See Also:
        - [L-moment - Wikipedia](https://wikipedia.org/wiki/L-moment)
        - [`scipy.stats.moment`][scipy.stats.moment]

    References:
        - [J.R.M. Hosking (1990)](https://jstor.org/stable/2345653)
        - [E. Elmamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    """
    # weight-adjusted $x_{i:n}$
    x_k = order_stats(
        a,
        axis=axis,
        dtype=dtype,
        fweights=fweights,
        aweights=aweights,
        sort=sort,
    )
    x_k = ensure_axis_at(x_k, axis, -1)

    r_max: int = clean_order(cast(
        int,
        np.max(np.asarray(r))  # pyright: ignore [reportUnknownMemberType]
    ))
    n = x_k.shape[-1]

    # projection matrix
    P_r = l_weights(r_max, n, trim, dtype=dtype)

    l_r = np.inner(P_r, x_k)

    # we like 0-based indexing; so if P_r starts at r=1, prepend all 1's
    # for r=0 (any zeroth moment is defined to be 1)
    l_r = np.r_[np.ones((1, *l_r.shape[1:]), dtype=l_r.dtype), l_r]

    assert np.all(l_r[0] == 1)
    assert len(l_r) == r_max + 1, (l_r.shape, r_max)

    # l[r] fails when r is e.g. a tuple (valid sequence).
    return l_r.take(r, 0)


def l_ratio(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> T | npt.NDArray[T]:
    """
    Estimates the generalized L-moment ratio:

    $$
    \\tau^{(t_1, t_2)}_{rs} = \\frac{
        \\lambda^{(t_1, t_2)}_r
    }{
        \\lambda^{(t_1, t_2)}_s
    }
    $$

    Equivalent to `lmo.l_moment(a, r, *, **) / lmo.l_moment(a, s, *, **)`.

    Notes:
        The L-moment with `r=0` is `1`, so the `l_ratio(a, r, 0, *, **)` is
        equivalent to `l_moment(a, r, *, **)`

    Examples:
        Estimate the L-location, L-scale, L-skewness and L-kurtosis
        simultaneously:

        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).lognormal(size=99)
        >>> lmo.l_ratio(x, [1, 2, 3, 4], [0, 0, 2, 2])
        array([1.53196368, 0.77549561, 0.4463163 , 0.29752178])
        >>> lmo.l_ratio(x, [1, 2, 3, 4], [0, 0, 2, 2], trim=(0, 1))
        array([0.75646807, 0.32203446, 0.23887609, 0.07917904])

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]

    """
    _r, _s = np.asarray(r), np.asarray(s)
    rs = np.stack(np.broadcast_arrays(_r, _s))

    l_rs = cast(
        npt.NDArray[T],
        l_moment(a, rs, trim, axis=axis, dtype=dtype, **kwargs)
    )

    r_eq_s = _r == _s
    with np.errstate(divide='ignore'):
        return np.where(
            r_eq_s,
            np.ones_like(l_rs[0]),
            np.divide(l_rs[0], l_rs[1], where=~r_eq_s)
        )[()]


def l_loc(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> T | npt.NDArray[T]:
    """
    *L-location* (or *L-loc*): unbiased estimator of the first L-moment,
    $\\lambda^{(t_1, t_2)}_1$.

    Alias for [`lmo.l_moment(a, 1, *, **)`][lmo.l_moment].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_cauchy(99)
        >>> x.mean()
        -7.564850346291148
        >>> lmo.l_loc(x)
        -7.56485034629115
        >>> lmo.l_loc(x, trim=(1, 1))
        -0.1592418019341137

    Notes:
        If `trim = (0, 0)` (default), the L-location is equivalent to the
        [arithmetic mean](https://wikipedia.org/wiki/Arithmetic_mean).

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
        - [`numpy.average`][numpy.average]

    """
    return l_moment(a, 1, trim, axis, dtype, **kwargs)


def l_scale(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> T | npt.NDArray[T]:
    """
    *L-scale*: unbiased estimator of the second L-moment,
    $\\lambda^{(t_1, t_2)}_2$

    Alias for [`lmo.l_moment(a, 2, *, **)`][lmo.l_moment].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_cauchy(99)
        >>> x.std()
        72.8771524453277
        >>> lmo.l_scale(x)
        9.501123995203486
        >>> lmo.l_scale(x, trim=(1, 1))
        0.6589932797072331

    Notes:
        If `trim = (0, 0)` (default), the L-scale is equivalent to half the
        [Gini mean difference (GMD)](
        https://wikipedia.org/wiki/Gini_mean_difference).

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
        - [`numpy.std`][numpy.std]

    """
    return l_moment(a, 2, trim, axis, dtype, **kwargs)


def l_variation(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> T | npt.NDArray[T]:
    """
    The *coefficient of L-variation* (or *L-CV*) unbiased sample estimator:

    $$
    \\tau^{(t_1, t_2)} = \\frac{
        \\lambda^{(t_1, t_2)}_2
    }{
        \\lambda^{(t_1, t_2)}_1
    }
    $$

    Alias for [`lmo.l_ratio(a, 2, 1, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).pareto(4.2, 99)
        >>> x.std() / x.mean()
        1.321611129357151
        >>> lmo.l_variation(x)
        0.5907363914181409
        >>> lmo.l_variation(x, trim=(0, 1))
        0.5539504469144685

    Notes:
        If `trim = (0, 0)` (default), this is equivalent to the
        [Gini coefficient](https://wikipedia.org/wiki/Arithmetic_mean),
        and lies within the interval $(0, 1)$.

    See Also:
        - [Gini coefficient - Wikipedia](
            https://wikipedia.org/wiki/Gini_coefficient)
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.variation.l_ratio`][scipy.stats.variation]

    """
    return l_ratio(a, 2, 1, trim, axis, dtype, **kwargs)


def l_skew(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> T | npt.NDArray[T]:
    """
    Unbiased sample estimator of the *coefficient of L-skewness*, or *L-skew*
    for short:

    $$
    \\tau^{(t_1, t_2)}_3
        = \\frac{
            \\lambda^{(t_1, t_2)}_3
        }{
            \\lambda^{(t_1, t_2)}_2
        }
    $$

    Alias for [`lmo.l_ratio(a, 3, 2, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_exponential(99)
        >>> lmo.l_skew(x)
        0.38524343888616047
        >>> lmo.l_skew(x, trim=(0, 1))
        0.2711613932338941

    See Also:
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.skew`][scipy.stats.skew]

    """
    return l_ratio(a, 3, 2, trim, axis, dtype, **kwargs)


def l_kurtosis(
    a: npt.ArrayLike,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> T | npt.NDArray[T]:
    """
    L-kurtosis coefficient; the 4th sample L-moment ratio.

    $$
    \\tau^{(t_1, t_2)}_4
        = \\frac{
            \\lambda^{(t_1, t_2)}_4
        }{
            \\lambda^{(t_1, t_2)}_2
        }
    $$

    Alias for [`lmo.l_ratio(a, 4, 2, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_t(2, 99)
        >>> lmo.l_kurtosis(x)
        0.28912787109746096
        >>> lmo.l_kurtosis(x, trim=(1, 1))
        0.19928182131774994

    Notes:
        The L-kurtosis $\\tau_4$ lies within the interval
        $[-\\frac{1}{4}, 1)$, and by the L-skewness $\\tau_3$ as
        $5 \\tau_3^2 - 1 \\le 4 \\tau_4$.

    See Also:
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.kurtosis`][scipy.stats.kurtosis]

    """
    return l_ratio(a, 4, 2, trim, axis, dtype, **kwargs)

