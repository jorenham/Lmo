"""Unbiased sample estimators of the generalized trimmed L-moments."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt

from . import ostats, pwm_beta
from ._utils import (
    clean_order,
    clean_orders,
    clean_trim,
    ensure_axis_at,
    l_stats_orders,
    moments_to_ratio,
    ordered,
    round0,
    sort_maybe,
)
from .linalg import ir_pascal, sandwich, sh_legendre, trim_matrix
from .typing import AnyOrder, AnyOrderND


if sys.version_info >= (3, 13):
    from typing import TypeVar, Unpack
else:
    from typing_extensions import TypeVar, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable

    import lmo.typing as lmt
    import lmo.typing.np as lnpt


__all__ = (
    'l_weights',

    'l_moment',
    'l_ratio',
    'l_stats',

    'l_loc',
    'l_scale',
    'l_variation',
    'l_skew',
    'l_kurtosis',
    'l_kurt',

    'l_moment_cov',
    'l_ratio_se',
    'l_stats_se',

    'l_moment_influence',
    'l_ratio_influence',
)


_OrderT = TypeVar('_OrderT', bound=int)
_SizeT = TypeVar('_SizeT', bound=int)
_SCT_f = TypeVar('_SCT_f', bound=np.floating[Any], default=np.float64)

_DType: TypeAlias = np.dtype[_SCT_f] | type[_SCT_f]
_Vectorized: TypeAlias = _SCT_f | npt.NDArray[_SCT_f]
_Floating: TypeAlias = np.floating[Any]

# (dtype.char, n, s, t)
_CacheKey: TypeAlias = (
    tuple[str, int, int, int]
    | tuple[str, int, float, float]
)
# `r: _T_order >= 4`
_CacheArray: TypeAlias = onpt.Array[tuple[_OrderT, _SizeT], np.longdouble]
_Cache: TypeAlias = dict[_CacheKey, _CacheArray[Any, Any]]

# depends on `dtype`, `n`, and `trim`
_CACHE: Final[_Cache] = {}


def _l_weights_pwm(
    r: _OrderT,
    n: _SizeT,
    /,
    trim: tuple[int, int],
    *,
    dtype: _DType[_SCT_f],
) -> onpt.Array[tuple[_OrderT, _SizeT], _SCT_f]:
    s, t = trim
    r0 = r + s + t

    # `__matmul__` annotations are lacking (`np.matmul` is equivalent to it)
    wr = np.matmul(
        sh_legendre(r0, dtype=np.int64 if r0 < 29 else dtype),
        pwm_beta.weights(r0, n, dtype=dtype),
        dtype=dtype,
    )
    if s or t:
        wr = np.matmul(trim_matrix(r, trim, dtype=dtype), wr, dtype=dtype)

        # ensure that the trimmed ends are 0
        if s:
            wr[:, :s] = 0
        if t:
            wr[:, -t:] = 0

    return wr


def _l_weights_ostat(
    r: _OrderT,
    n: _SizeT,
    /,
    trim: tuple[int, int] | tuple[float, float],
    *,
    dtype: _DType[_SCT_f],
) -> onpt.Array[tuple[_OrderT, _SizeT], _SCT_f]:
    assert r >= 1, r

    s, t = trim
    assert 0 < r + s + t <= n, (r, n, trim)
    assert s >= 0, trim
    assert t >= 0, trim

    c = ir_pascal(r, dtype=dtype)
    jnj = np.arange(n, dtype=dtype)
    jnj /= n - jnj

    out = np.zeros((r, n), dtype=dtype)
    for j in range(r):
        w0 = ostats.weights(s, s + t + j + 1, n)
        out[j] = c[j, 0] * w0
        for k in range(1, j + 1):
            # order statistic recurrence relation
            w0 = np.roll(w0, 1) * jnj * ((t + j - k + 1) / (s + k))
            out[j] += c[j, k] * w0
    return out


def l_weights(
    r_max: _OrderT,
    n: _SizeT,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    dtype: _DType[_SCT_f] = np.float64,
    cache: bool | None = None,
) -> onpt.Array[tuple[_OrderT, _SizeT], _SCT_f]:
    r"""
    Projection matrix of the first $r$ (T)L-moments for $n$ samples.

    For integer trim is the matrix is a linear combination of the Power
    Weighted Moment (PWM) weights (the sample estimator of $\beta_{r_1}$), and
    the shifted Legendre polynomials.

    If the trimmings are nonzero and integers, a linearized (and corrected)
    adaptation of the recurrence relations from *Hosking (2007)* are applied,
    as well.

    $$
    (2k + s + t - 1) \lambda^{(s, t)}_k
        = (k + s + t) \lambda^{(s - 1, t)}_k
        + \frac{1}{k} (k + 1) (k + t) \lambda^{(s - 1, t)}_{k+1}
    $$

    for $s > 0$, and

    $$
    (2k + s + t - 1) \lambda^{(s, t)}_k
        = (k + s + t) \lambda^{(s, t - 1)}_k
        - \frac{1}{k} (k + 1) (k + s) \lambda^{(s, t - 1)}_{k+1}
    $$

    for $t > 0$.

    If the trim values are floats instead, the weights are calculated directly
    from the (generalized) order statistics. At the time of writing (07-2023),
    these "generalized trimmed L-moments" have not been discussed in the
    literature or the R-packages. It's probably a good idea to publish this...

    TLDR:
        This matrix (linearly) transforms $x_{i:n}$ (i.e. the sorted
        observation vector(s) of size $n$), into (an unbiased estimate of) the
        *generalized trimmed L-moments*, with orders $\le r$.

    Args:
        r_max: The amount L-moment orders.
        n: The number of samples.
        trim: A scalar or 2-tuple with the trim orders. Defaults to 0.
        dtype: The datatype of the returned weight matrix.
        cache: Whether to cache the weights. By default, it's enabled i.f.f
            the trim values are integers, and `r_max + sum(trim) < 24`.

    Returns:
        P_r: 2-D array of shape `(r_max, n)`, readonly if `cache=True`

    Examples:
        >>> import lmo
        >>> lmo.l_weights(3, 4)
        array([[ 0.25      ,  0.25      ,  0.25      ,  0.25      ],
               [-0.25      , -0.08333333,  0.08333333,  0.25      ],
               [ 0.25      , -0.25      , -0.25      ,  0.25      ]])
        >>> _ @ [-1, 0, 1 / 2, 3 / 2]
        array([0.25      , 0.66666667, 0.        ])

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
    """
    if r_max < 0:
        msg = f'r must be non-negative, got {r_max}'
        raise ValueError(msg)

    dtype = np.dtype(dtype)
    sctype = dtype.type

    if r_max == 0:
        return np.empty((0, n), dtype=sctype)

    s, t = clean_trim(trim)

    if (n_min := r_max + s + t) > n:
        msg = f'expected n >= r + s + t, got {n} < {n_min}'
        raise ValueError(msg)

    key = dtype.char, n, s, t
    if (_w := _CACHE.get(key)) is not None and _w.shape[0] >= r_max:
        w = cast(onpt.Array[tuple[_OrderT, _SizeT], _SCT_f], _w)
    else:
        # when caching, use at least 4 orders, to avoid cache misses
        _r_max = 4 if cache and r_max < 4 else r_max

        _cache_default = False
        if r_max + s + t <= 24 and isinstance(s, int) and isinstance(t, int):
            w = _l_weights_pwm(_r_max, n, trim=(s, t), dtype=sctype)
            _cache_default = True
        else:
            w = _l_weights_ostat(_r_max, n, trim=(s, t), dtype=sctype)

        if cache or cache is None and _cache_default:
            w.setflags(write=False)
            # be wary of a potential race condition
            if key not in _CACHE or w.shape[0] >= _CACHE[key].shape[0]:
                _CACHE[key] = w

    if w.shape[0] > r_max:
        w = w[:r_max]
    return w


@overload
def l_moment(
    a: lnpt.AnyArrayFloat,
    r: AnyOrder,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
@overload
def l_moment(
    a: lnpt.AnyMatrixFloat | lnpt.AnyTensorFloat,
    r: AnyOrder | AnyOrderND,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]: ...
@overload
def l_moment(
    a: lnpt.AnyVectorFloat,
    r: AnyOrder,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
@overload
def l_moment(
    a: lnpt.AnyArrayFloat,
    r: AnyOrderND,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]: ...
def l_moment(
    a: lnpt.AnyArrayFloat,
    r: AnyOrder | AnyOrderND,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    fweights: lmt.AnyFWeights | None = None,
    aweights: lmt.AnyAWeights | None = None,
    sort: lnpt.SortKind | bool = True,
    cache: bool | None = None,
) -> _Vectorized[_SCT_f]:
    r"""
    Estimates the generalized trimmed L-moment $\lambda^{(s, t)}_r$ from
    the samples along the specified axis. By default, this will be the regular
    L-moment, $\lambda_r = \lambda^{(0, 0)}_r$.

    Parameters:
        a:
            Array containing numbers whose L-moments is desired.
            If `a` is not an array, a conversion is attempted.
        r:
            The L-moment order(s), non-negative integer or array.
        trim:
            Left- and right-trim orders $(s, t)$, non-negative ints or
            floats that are bound by $s + t < n - r$.
            A single scalar $t$ can be proivided as well, as alias for
            $(t, t)$.

            Some special cases include:

            - $(0, 0)$ or $(0)$: The original **L**-moment, introduced by
                Hosking in 1990.
            - $(t, t)$ or $(t)$: **TL**-moment (**T**rimmed L-moment)
                $\lambda_r^{(t)}$, with symmetric trimming. First introduced by
                Elamir & Seheult in 2003, and refined by Hosking in 2007.
                Generally more robust than L-moments. Useful for fitting
                pathological distributions, such as the Cauchy distribution.
            - $(0, t)$: **LL**-moment (**L**inear combination of **L**owest
                order statistics), introduced by Bayazit & Onoz in 2002.
                Assigns more weight to smaller observations.
            - $(s, 0)$: **LH**-moment (**L**inear combination of **H**igher
                order statistics), as described by Wang in 1997.
                Assigns more weight to larger observations.
        axis:
            Axis along which to calculate the moments. If `None` (default),
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
        sort (True | False | 'quick' | 'heap' | 'stable'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].
            Set to `False` if the array is already sorted.
        cache:
            Set to `True` to speed up future L-moment calculations that have
            the same number of observations in `a`, equal `trim`, and equal or
            smaller `r`. By default, it will cache i.f.f. the trim is integral,
            and $r + s + t \le 24$. Set to `False` to always disable caching.

    Returns:
        l:
            The L-moment(s) of the input This is a scalar iff a is 1-d and
            r is a scalar. Otherwise, this is an array with
            `np.ndim(r) + np.ndim(a) - 1` dimensions and shape like
            `(*np.shape(r), *(d for d in np.shape(a) if d != axis))`.

    Examples:
        Calculate the L-location and L-scale from student-T(2) samples, for
        different (symmetric) trim-lengths.

        >>> import lmo
        >>> import numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.standard_t(2, 99)

        >>> lmo.l_moment(x, [1, 2])
        array([-0.01412282,  0.94063132])
        >>> lmo.l_moment(x, [1, 2], trim=1)
        array([-0.0124483 ,  0.40120115])
        >>> lmo.l_moment(x, [1, 2], trim=(1, 1))
        array([-0.0124483 ,  0.40120115])

        The theoretical L- and TL-location is `0`, the L-scale is `1.1107`,
        and the TL-scale is `0.4165`, respectively.

    See Also:
        - [L-moment - Wikipedia](https://wikipedia.org/wiki/L-moment)
        - [`scipy.stats.moment`][scipy.stats.moment]

    References:
        - [J.R.M. Hosking (1990)](https://jstor.org/stable/2345653)
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
    """
    x_k = ordered(
        a,
        axis=axis,
        dtype=dtype,
        fweights=fweights,
        aweights=aweights,
        sort=sort,
    )
    x_k = ensure_axis_at(x_k, axis, -1)
    n = x_k.shape[-1]

    if np.isscalar(r):
        _r = np.array(clean_order(cast(AnyOrder, r)))
    else:
        _r = clean_orders(cast(AnyOrderND, r))
    r_min, r_max = np.min(_r), int(np.max(_r))

    # TODO @jorenham: nan handling, see:
    # https://github.com/jorenham/Lmo/issues/70

    (s, t) = st = clean_trim(trim)

    # ensure that any inf's (not nan's) are properly trimmed
    if (s or t) and isinstance(s, int):
        if s:
            x_k[..., :s] = np.nan_to_num(x_k[..., :s], nan=np.nan)
        if t:
            x_k[..., -t:] = np.nan_to_num(x_k[..., -t:], nan=np.nan)

    l_r = np.inner(l_weights(r_max, n, st, dtype=dtype, cache=cache), x_k)

    if r_min > 0:
        return l_r.take(_r - 1, 0)

    return np.r_[np.ones((1, *l_r.shape[1:]), l_r.dtype), l_r].take(_r, 0)


@overload
def l_ratio(
    a: lnpt.AnyVectorFloat,
    r: AnyOrder,
    s: AnyOrder,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int | None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
@overload
def l_ratio(
    a: lnpt.AnyMatrixFloat | lnpt.AnyTensorFloat,
    r: AnyOrder,
    s: AnyOrder,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]: ...
@overload
def l_ratio(
    a: lnpt.AnyArrayFloat,
    r: AnyOrder,
    s: AnyOrder,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
@overload
def l_ratio(
    a: lnpt.AnyArrayFloat,
    r: AnyOrder,
    s: AnyOrderND,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]: ...
@overload
def l_ratio(
    a: lnpt.AnyArrayFloat,
    r: AnyOrderND,
    s: AnyOrder | AnyOrderND,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]: ...
def l_ratio(
    a: lnpt.AnyArrayFloat,
    r: AnyOrder | AnyOrderND,
    s: AnyOrder | AnyOrderND,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _Vectorized[_SCT_f]:
    r"""
    Estimates the generalized L-moment ratio.

    $$
    \tau^{(s, t)}_{rs} = \frac
        {\lambda^{(s, t)}_r}
        {\lambda^{(s, t)}_s}
    $$

    Equivalent to `lmo.l_moment(a, r, *, **) / lmo.l_moment(a, s, *, **)`.

    The L-moment with `r=0` is `1`, so the `l_ratio(a, r, 0, *, **)` is
    equivalent to `l_moment(a, r, *, **)`.

    Notes:
        Often, when referring to the $r$th *L-ratio*, the L-moment ratio with
        $k=2$ is implied, i.e. $\tau^{(s, t)}_r$ is short-hand notation
        for $\tau^{(s, t)}_{r,2}$.

        The L-variation (L-moment Coefficient of Variation, or L-CB) is
        another special case of the L-moment ratio, $\tau^{(s, t)}_{2,1}$.
        It is sometimes denoted in the literature by dropping the subscript
        indices: $\tau^{(s, t)}$.
        Note that this should only be used with strictly positive
        distributions.

    Examples:
        Estimate the L-location, L-scale, L-skewness and L-kurtosis
        simultaneously:

        >>> import lmo
        >>> import numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.lognormal(size=99)
        >>> lmo.l_ratio(x, [1, 2, 3, 4], [0, 0, 2, 2])
        array([1.53196368, 0.77549561, 0.4463163 , 0.29752178])
        >>> lmo.l_ratio(x, [1, 2, 3, 4], [0, 0, 2, 2], trim=(0, 1))
        array([0.75646807, 0.32203446, 0.23887609, 0.07917904])

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
    """
    rs = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))
    l_rs = l_moment(a, rs, trim=trim, axis=axis, dtype=dtype, **kwds)
    return moments_to_ratio(rs, l_rs)


def l_stats(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = 0,
    num: int = 4,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]:
    """
    Calculates the L-loc(ation), L-scale, L-skew(ness) and L-kurtosis.

    Equivalent to `lmo.l_ratio(a, [1, 2, 3, 4], [0, 0, 2, 2], *, **)` by
    default.

    Examples:
        >>> import lmo, scipy.stats
        >>> x = scipy.stats.gumbel_r.rvs(size=99, random_state=12345)
        >>> lmo.l_stats(x)
        array([0.79014773, 0.68346357, 0.12207413, 0.12829047])

        The theoretical L-stats of the standard Gumbel distribution are
        `[0.577, 0.693, 0.170, 0.150]`.

    See Also:
        - [`lmo.l_stats_se`][lmo.l_stats_se]
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`lmo.l_costats`][lmo.l_costats]
    """
    r, s = l_stats_orders(num)
    return l_ratio(a, r, s, trim=trim, axis=axis, dtype=dtype, **kwds)


@overload
def l_loc(
    a: lnpt.AnyMatrixFloat | lnpt.AnyTensorFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]: ...
@overload
def l_loc(
    a: lnpt.AnyVectorFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
@overload
def l_loc(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
def l_loc(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _Vectorized[_SCT_f]:
    r"""
    *L-location* (or *L-loc*): unbiased estimator of the first L-moment,
    $\lambda^{(s, t)}_1$.

    Alias for [`lmo.l_moment(a, 1, *, **)`][lmo.l_moment].

    Examples:
        The first moment (i.e. the mean) of the Cauchy distribution does not
        exist. This means that estimating the location of a Cauchy
        distribution from its samples, cannot be done using the traditional
        average (i.e. the arithmetic mean).
        Instead, a robust location measure should be used, e.g. the median,
        or the TL-location.

        To illustrate, let's start by drawing some samples from the standard
        Cauchy distribution, which is centered around the origin.

        >>> import lmo
        >>> import numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.standard_cauchy(200)

        The mean and the untrimmed L-location (which are equivalent) give
        wrong results, so don't do this:

        >>> np.mean(x)
        -3.6805
        >>> lmo.l_loc(x)
        -3.6805

        Usually, the answer to this problem is to use the median.
        However, the median only considers one or two samples (depending on
        whether the amount of samples is odd or even, respectively).
        So the median ignores most of the available information.

        >>> np.median(x)
        0.096825
        >>> lmo.l_loc(x, trim=(len(x) - 1) // 2)
        0.096825

        Luckily for us, Lmo knows how to deal with longs tails, as well --
        trimming them (specifically, by skipping the first $s$ and last $t$
        expected order statistics).

        Let's try the TL-location (which is equivalent to the median)

        >>> lmo.l_loc(x, trim=1)  # equivalent to `trim=(1, 1)`
        0.06522

    Notes:
        The trimmed L-location naturally unifies the arithmetic mean, the
        median, the minimum and the maximum. In particular, the following are
        equivalent, given `n = len(x)`:

        - `l_loc(x, trim=0)` / `statistics.mean(x)` / `np.mean(x)`
        - `l_loc(x, trim=(n-1) // 2)` / `statistics.median(x)` / `np.median(x)`
        - `l_loc(x, trim=(0, n-1))` / `min(x)` / `np.min(x)`
        - `l_loc(x, trim=(n-1, 0))` / `max(x)` / `np.max(x)`

        Note that numerical noise might cause slight differences between
        their results.

        Even though `lmo` is built with performance in mind, the equivalent
        `numpy` functions are always faster, as they don't need to
        sort *all* samples. Specifically, the time complexity of `lmo.l_loc`
        (and `l_moment` in general) is $O(n \log n)$, whereas that of
        `numpy.{mean,median,min,max}` is `O(n)`

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
        - [`numpy.average`][numpy.average]
    """
    return l_moment(a, 1, trim=trim, axis=axis, dtype=dtype, **kwds)


@overload
def l_scale(
    a: lnpt.AnyMatrixFloat | lnpt.AnyTensorFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]: ...
@overload
def l_scale(
    a: lnpt.AnyVectorFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
@overload
def l_scale(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
def l_scale(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _Vectorized[_SCT_f]:
    r"""
    *L-scale* unbiased estimator for the second L-moment,
    $\lambda^{(s, t)}_2$.

    Alias for [`lmo.l_moment(a, 2, *, **)`][lmo.l_moment].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_cauchy(99)
        >>> x.std()
        72.87715244
        >>> lmo.l_scale(x)
        9.501123995
        >>> lmo.l_scale(x, trim=(1, 1))
        0.658993279

    Notes:
        If `trim = (0, 0)` (default), the L-scale is equivalent to half the
        [Gini mean difference (GMD)](
        https://wikipedia.org/wiki/Gini_mean_difference).

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
        - [`numpy.std`][numpy.std]
    """
    return l_moment(a, 2, trim=trim, axis=axis, dtype=dtype, **kwds)


@overload
def l_variation(
    a: lnpt.AnyMatrixFloat | lnpt.AnyTensorFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> onpt.Array[Any, _SCT_f]: ...
@overload
def l_variation(
    a: lnpt.AnyVectorFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: int,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
@overload
def l_variation(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _SCT_f: ...
def l_variation(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _Vectorized[_SCT_f]:
    r"""
    The *coefficient of L-variation* (or *L-CV*) unbiased sample estimator:

    $$
    \tau^{(s, t)} = \frac
        {\lambda^{(s, t)}_2}
        {\lambda^{(s, t)}_1}
    $$

    Alias for [`lmo.l_ratio(a, 2, 1, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).pareto(4.2, 99)
        >>> x.std() / x.mean()
        1.32161112
        >>> lmo.l_variation(x)
        0.59073639
        >>> lmo.l_variation(x, trim=(0, 1))
        0.55395044

    Notes:
        If `trim = (0, 0)` (default), this is equivalent to the
        [Gini coefficient](https://wikipedia.org/wiki/Arithmetic_mean),
        and lies within the interval $(0, 1)$.

    See Also:
        - [Gini coefficient - Wikipedia](
            https://wikipedia.org/wiki/Gini_coefficient)
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.variation.l_ratio`][scipy.stats.variation]
    """  # noqa: D415
    return l_ratio(a, 2, 1, trim=trim, axis=axis, dtype=dtype, **kwds)


def l_skew(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _Vectorized[_SCT_f]:
    r"""
    Unbiased sample estimator for the *L-skewness* coefficient.

    $$
    \tau^{(s, t)}_3 = \frac
        {\lambda^{(s, t)}_3}
        {\lambda^{(s, t)}_2}
    $$

    Alias for [`lmo.l_ratio(a, 3, 2, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_exponential(99)
        >>> lmo.l_skew(x)
        0.38524343
        >>> lmo.l_skew(x, trim=(0, 1))
        0.27116139

    See Also:
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.skew`][scipy.stats.skew]
    """
    return l_ratio(a, 3, 2, trim=trim, axis=axis, dtype=dtype, **kwds)


def l_kurtosis(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _Vectorized[_SCT_f]:
    r"""
    L-kurtosis coefficient; the 4th sample L-moment ratio.

    $$
    \tau^{(s, t)}_4 = \frac
        {\lambda^{(s, t)}_4}
        {\lambda^{(s, t)}_2}
    $$

    Alias for [`lmo.l_ratio(a, 4, 2, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_t(2, 99)
        >>> lmo.l_kurtosis(x)
        0.28912787
        >>> lmo.l_kurtosis(x, trim=(1, 1))
        0.19928182

    Notes:
        The L-kurtosis $\tau_4$ lies within the interval
        $[-\frac{1}{4}, 1)$, and by the L-skewness $\\tau_3$ as
        $5 \tau_3^2 - 1 \le 4 \tau_4$.

    See Also:
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.kurtosis`][scipy.stats.kurtosis]
    """
    return l_ratio(a, 4, 2, trim=trim, axis=axis, dtype=dtype, **kwds)


l_kurt = l_kurtosis
"""Alias for [`lmo.l_kurtosis`][lmo.l_kurtosis]."""


def l_moment_cov(
    a: lnpt.AnyArrayFloat,
    r_max: AnyOrder,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Any,
) -> npt.NDArray[_SCT_f]:
    """
    Non-parmateric auto-covariance matrix of the generalized trimmed
    L-moment point estimates with orders `r = 1, ..., r_max`.

    Returns:
        S_l: Variance-covariance matrix/tensor of shape `(r_max, r_max, ...)`

    Examples:
        Fitting of the cauchy distribution with TL-moments. The location is
        equal to the TL-location, and scale should be $0.698$ times the
        TL(1)-scale, see Elamir & Seheult (2003).

        >>> import lmo, numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.standard_cauchy(1337)
        >>> lmo.l_moment(x, [1, 2], trim=(1, 1))
        array([0.08142405, 0.68884917])

        The L-moment estimates seem to make sense. Let's check their standard
        errors, by taking the square root of the variances (the diagonal of the
        covariance matrix):

        >>> lmo.l_moment_cov(x, 2, trim=(1, 1))
        array([[ 4.89407076e-03, -4.26419310e-05],
               [-4.26419310e-05,  1.30898414e-03]])
        >>> np.sqrt(_.diagonal())
        array([0.06995764, 0.03617989])

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
        - [Covariance matrix - Wikipedia](
            https://wikipedia.org/wiki/Covariance_matrix)

    References:
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [E. Elamir & A. Seheult (2004) - Exact variance structure of sample
            L-moments](https://doi.org/10.1016/S0378-3758(03)00213-1)

    Todo:
        - Use the direct (Jacobi) method from Hosking (2015).
    """
    _r_max = clean_order(r_max, 'r_max')
    _trim = cast(tuple[int, int], clean_trim(trim))

    if any(int(t) != t for t in _trim):
        msg = 'l_moment_cov does not support fractional trimming (yet)'
        raise TypeError(msg)

    ks = _r_max + sum(_trim)
    if ks < _r_max:
        msg = 'trimmings must be positive'
        raise ValueError(msg)

    # projection matrix: PWMs -> generalized trimmed L-moments
    p_l: npt.NDArray[np.floating[Any]]
    p_l = trim_matrix(_r_max, trim=_trim, dtype=dtype) @ sh_legendre(ks)
    # clean some numerical noise
    # p_l = np.round(p_l, 12) + 0.

    # PWM covariance matrix
    s_b = pwm_beta.cov(a, ks, axis=axis, dtype=dtype, **kwds)

    # tasty, eh?
    return sandwich(p_l, s_b, dtype=dtype)


def l_ratio_se(
    a: lnpt.AnyArrayFloat,
    r: AnyOrder | AnyOrderND,
    s: AnyOrder | AnyOrderND,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _Vectorized[_SCT_f]:
    """
    Non-parametric estimates of the Standard Error (SE) in the L-ratio
    estimates from [`lmo.l_ratio`][lmo.l_ratio].

    Examples:
        Estimate the values and errors of the TL-loc, scale, skew and kurtosis
        for Cauchy-distributed samples. The theoretical values are
        `[0.0, 0.698, 0.0, 0.343]` (Elamir & Seheult, 2003), respectively.

        >>> import lmo, numpy as np
        >>> rng = np.random.default_rng(12345)
        >>> x = rng.standard_cauchy(42)
        >>> lmo.l_ratio(x, [1, 2, 3, 4], [0, 0, 2, 2], trim=(1, 1))
        array([-0.25830513,  0.61738638, -0.03069701,  0.25550176])
        >>> lmo.l_ratio_se(x, [1, 2, 3, 4], [0, 0, 2, 2], trim=(1, 1))
        array([0.32857302, 0.12896501, 0.13835403, 0.07188138])

    See Also:
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`lmo.l_moment_cov`][lmo.l_moment_cov]
        - [Propagation of uncertainty](
            https://wikipedia.org/wiki/Propagation_of_uncertainty)

    References:
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [E. Elamir & A. Seheult (2004) - Exact variance structure of sample
            L-moments](https://doi.org/10.1016/S0378-3758(03)00213-1)

    """
    _r, _s = np.broadcast_arrays(np.asarray(r), np.asarray(s))
    _rs = np.stack((_r, _s))
    r_max = np.amax(np.r_[_r, _s].ravel())

    # L-moments
    l_rs = l_moment(a, _rs, trim, axis=axis, dtype=dtype, **kwds)
    l_r, l_s = l_rs[0], l_rs[1]

    # L-moment auto-covariance matrix
    k_l = l_moment_cov(a, r_max, trim, axis=axis, dtype=dtype, **kwds)
    # prepend the "zeroth" moment, with has 0 (co)variance
    k_l = np.pad(k_l, (1, 0), constant_values=0)

    s_rr = k_l[_r, _r]  # Var[l_r]
    s_ss = k_l[_s, _s]  # Var[l_r]
    s_rs = k_l[_r, _s]  # Cov[l_r, l_s]

    # the classic approximation to propagation of uncertainty for an RV ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        _s_tt = (l_r / l_s) ** 2 * (
            s_rr / l_r**2 + s_ss / l_s**2 - 2 * s_rs / (l_r * l_s)
        )
        # Var[l_r / l_r] = Var[1] = 0
        _s_tt = np.where(_s == 0, s_rr, _s_tt)
        # Var[l_r / l_0] == Var[l_r / 1] == Var[l_r]
        s_tt = np.where(_r == _s, 0, _s_tt)

    return np.sqrt(s_tt)


def l_stats_se(
    a: lnpt.AnyArrayFloat,
    /,
    trim: lmt.AnyTrim = 0,
    num: int = 4,
    *,
    axis: int | None = None,
    dtype: _DType[_SCT_f] = np.float64,
    **kwds: Unpack[lmt.LMomentOptions],
) -> _Vectorized[_SCT_f]:
    """
    Calculates the standard errors (SE's) of the [`L-stats`][lmo.l_stats].

    Equivalent to `lmo.l_ratio_se(a, [1, 2, 3, 4], [0, 0, 2, 2], *, **)` by
    default.

    Examples:
        >>> import lmo, scipy.stats
        >>> x = scipy.stats.gumbel_r.rvs(size=99, random_state=12345)
        >>> lmo.l_stats(x)
        array([0.79014773, 0.68346357, 0.12207413, 0.12829047])
        >>> lmo.l_stats_se(x)
        array([0.12305147, 0.05348839, 0.04472984, 0.03408495])

        The theoretical L-stats of the standard Gumbel distribution are
        `[0.577, 0.693, 0.170, 0.150]`. The corresponding relative z-scores
        are `[-1.730, 0.181, 1.070, 0.648]`.

    See Also:
        - [`lmo.l_stats`][lmo.l_stats]
        - [`lmo.l_ratio_se`][lmo.l_ratio_se]
    """
    r, s = l_stats_orders(num)
    return l_ratio_se(a, r, s, trim=trim, axis=axis, dtype=dtype, **kwds)


_T_x = TypeVar('_T_x', bound=float | npt.NDArray[_Floating])


def l_moment_influence(
    a: lnpt.AnyVectorFloat,
    r: AnyOrder,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    sort: lnpt.SortKind | bool = True,
    tol: float = 1e-8,
) -> Callable[[_T_x], _T_x]:
    r"""
    Calculate the *Empirical Influence Function (EIF)* for a
    [sample L-moment][lmo.l_moment] estimate.

    Notes:
        This function is *not* vectorized, and can only be used for a single
        L-moment order `r`.
        However, the returned (empirical influence) function *is* vectorized.

    Args:
        a: 1-D array-like containing observed samples.
        r: L-moment order. Must be a non-negative integer.
        trim:
            Left- and right- trim. Can be scalar or 2-tuple of
            non-negative int (or float).

    Other parameters:
        sort (True | False | 'quick' | 'heap' | 'stable'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].
            Set to `False` if the array is already sorted.
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The (vectorized) empirical influence function.

    Raises:
        ValueError: If `a` is not 1-D array-like.
        TypeError: If `a` is not a floating-point type.

    """
    _r = clean_order(r)
    s, t = clean_trim(trim)

    x_k = np.array(a, copy=bool(sort), dtype=np.float64)
    if x_k.ndim != 1:
        msg = f"'a' must be 1-D array-like, got ndim={x_k.ndim}."
        raise ValueError(msg)
    x_k = sort_maybe(x_k, sort=sort, inplace=True)

    n = len(x_k)

    w_k: onpt.Array[tuple[int], np.float64] = l_weights(_r, n, (s, t))[-1]
    l_r = cast(np.float64, w_k @ x_k)

    def influence_function(x: _T_x, /) -> _T_x:
        _x = np.asanyarray(x)

        # ECDF
        # k = np.maximum(np.searchsorted(x_k, _x, side='right') - 1, 0)
        w = np.interp(
            _x,
            x_k,
            w_k,
            left=0 if s else w_k[0],
            right=0 if t else w_k[-1],
        )
        alpha = n * w * np.where(w, _x, 0)
        out = round0(alpha - l_r, tol=tol)

        if _x.ndim == 0 and np.isscalar(x):
            return out.item()
        return cast(_T_x, out)

    influence_function.__doc__ = (
        f'Empirical L-moment influence function for {r=}, {trim=}, and {n=}.'
    )
    # piggyback the L-moment, to avoid recomputing it in l_ratio_influence
    influence_function.l = l_r  # pyright: ignore[reportFunctionMemberAccess]
    return influence_function


def l_ratio_influence(
    a: lnpt.AnyVectorFloat,
    r: AnyOrder,
    s: AnyOrder = 2,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    sort: lnpt.SortKind | bool = True,
    tol: float = 1e-8,
) -> Callable[[_T_x], _T_x]:
    r"""
    Calculate the *Empirical Influence Function (EIF)* for a
    [sample L-moment ratio][lmo.l_ratio] estimate.

    Notes:
        This function is *not* vectorized, and can only be used for a single
        L-moment order `r`.
        However, the returned (empirical influence) function *is* vectorized.

    Args:
        a: 1-D array-like containing observed samples.
        r: L-moment ratio order. Must be a non-negative integer.
        s: Denominator L-moment order, defaults to 2.
        trim:
            Left- and right- trim. Can be scalar or 2-tuple of
            non-negative int or float.

    Other parameters:
        sort (True | False | 'quick' | 'heap' | 'stable'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].
            Set to `False` if the array is already sorted.
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The (vectorized) empirical influence function.

    """
    _r, _s = clean_order(r), clean_order(s, name='s')

    _x = np.array(a, copy=bool(sort))
    _x = sort_maybe(_x, sort=sort, inplace=True)
    n = len(_x)

    eif_r = l_moment_influence(_x, _r, trim, sort=False, tol=0)
    eif_k = l_moment_influence(_x, _s, trim, sort=False, tol=0)

    l_r, l_k = cast(
        tuple[float, float],
        (eif_r.l, eif_k.l),  # pyright: ignore[reportFunctionMemberAccess]
    )
    if abs(l_k) <= tol * abs(l_r):
        msg = f'L-ratio ({r=}, {s=}) denominator is approximately zero.'
        raise ZeroDivisionError(msg)

    t_r = l_r / l_k

    def influence_function(x: _T_x, /) -> _T_x:
        psi_r = eif_r(x)
        # cheat a bit to avoid `inf - inf = nan` situations
        psi_k = np.where(np.isinf(psi_r), 0, eif_k(x))

        return cast(_T_x, round0((psi_r - t_r * psi_k) / l_k, tol=tol)[()])

    influence_function.__doc__ = (
        f'Theoretical influence function for L-moment ratio with r={_r}, '
        f'k={_s}, {trim=}, and {n=}'
    )
    return influence_function
