"""Unbiased sample estimators of the generalized trimmed L-moments."""

__all__ = (
    'l_weights',

    'l_loc',
    'l_scale',
    'l_variation',
    'l_skew',
    'l_kurtosis',

    'l_moment',
    'l_ratio',
    'l_stats',

    'l_moment_cov',
    'l_ratio_se',
    'l_stats_se',

    'l_moment_influence',
    'l_ratio_influence',
)

import sys
from collections.abc import Callable
from typing import Any, Final, SupportsIndex, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt

from . import ostats, pwm_beta
from ._utils import (
    clean_order,
    clean_trim,
    ensure_axis_at,
    l_stats_orders,
    moments_to_ratio,
    ordered,
    round0,
)
from .linalg import ir_pascal, sandwich, sh_legendre, trim_matrix
from .typing import AnyInt, AnyTrim, IntVector, LMomentOptions, SortKind

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack

T = TypeVar('T', bound=np.floating[Any])
V = TypeVar('V', bound=float | npt.NDArray[np.floating[Any]])


# Low-level weight methods

_L_WEIGHTS_CACHE: Final[
    dict[
        tuple[int, int | float, int | float],  # (n, s, t)
        npt.NDArray[np.floating[Any]],
    ]
] = {}


def _l_weights_pwm(
    r: int,
    n: int,
    /,
    trim: tuple[int, int],
    dtype: np.dtype[T] | type[T] = np.float64,
) -> npt.NDArray[T]:
    s, t = trim
    r0 = r + s + t

    p0 = sh_legendre(r0, dtype=np.int64 if r0 < 29 else dtype)
    w0 = p0 @ pwm_beta.weights(r0, n, dtype=dtype)  # type: ignore
    out = trim_matrix(r, trim, dtype=dtype) @ w0 if s or t else w0
    return cast(npt.NDArray[T], out)

    # remove numerical noise from the trimmings, and correct for potential
    # shifts in means
    # p_r[:, :t1] = p_r[:, n - t2:] = 0
    # p_r[1:, t1:n - t2] -= p_r[1:, t1:n - t2].mean(1, keepdims=True)

    # return p_r


def _l_weights_ostat(
    r: int,
    N: int,  # noqa: N803
    /,
    trim: tuple[float, float],
    dtype: np.dtype[T] | type[T] = np.float64,
) -> npt.NDArray[T]:
    s, t = trim

    assert 0 < r + s + t <= N, (r, N, trim)
    assert r >= 1, r
    assert s >= 0 and t >= 0, trim

    c = ir_pascal(r, dtype=dtype)
    jnj = np.arange(N, dtype=dtype)
    jnj /= N - jnj

    out = np.zeros((r, N), dtype=dtype)
    for n in range(r):
        w0 = ostats.weights(s, s + t + n + 1, N)
        out[n] = c[n, 0] * w0
        for k in range(1, n + 1):
            # order statistic recurrence relation
            w0 = np.roll(w0, 1) * jnj * ((t + n - k + 1) / (s + k))
            out[n] += c[n, k] * w0
    return out


def l_weights(
    r: int,
    n: int,
    /,
    trim: AnyTrim = (0, 0),
    dtype: np.dtype[T] | type[T] = np.float64,
    *,
    cache: bool = False,
) -> npt.NDArray[T]:
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

    Returns:
        P_r: 2-D array of shape `(r, n)`.

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
    if r == 0:
        return np.empty((r, n), dtype=dtype)

    match clean_trim(trim):
        case s, t if s < 0 or t < 0:
            msg = f'trim orders must be >=0, got {trim}'
            raise ValueError(msg)
        case s, t:
            pass
        case _:  # type: ignore [reportUneccessaryComparison]
            msg = (
                f'trim must be a tuple with two non-negative ints or floats, '
                f'got {trim!r}'
            )
            raise TypeError(msg)

    # manual cache lookup, only if cache=False (for testability)
    # e.g. `functools.cache` would be inefficient for e.g. r=3 with cached r=4
    cache_key = n, s, t
    if (
        cache
        and cache_key in _L_WEIGHTS_CACHE
        and (w := _L_WEIGHTS_CACHE[cache_key]).shape[0] <= r
    ):
        if w.shape[0] < r:
            w = w[:r]

        # ignore if r is larger that what's cached
        if w.shape[0] == r:
            assert w.shape == (r, n)
            return w.astype(dtype)

    if isinstance(s, int | np.integer) and isinstance(t, int | np.integer):
        w = _l_weights_pwm(r, n, trim=(int(s), int(t)), dtype=dtype)

        # ensure that the trimmed ends are 0
        if s:
            w[:, :s] = 0
        if t:
            w[:, -t:] = 0
    else:
        w = _l_weights_ostat(r, n, trim=(float(s), float(t)), dtype=dtype)

    if cache:
        # memoize
        _L_WEIGHTS_CACHE[cache_key] = w

    return w


# Summary statistics


@overload
def l_moment(
    a: npt.ArrayLike,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: type[np.float64] = ...,
    fweights: IntVector | None = ...,
    aweights: npt.ArrayLike | None = ...,
    sort: SortKind | None = ...,
    cache: bool = ...,
) -> np.float64:
    ...


@overload
def l_moment(
    a: npt.ArrayLike,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: np.dtype[T] | type[T],
    fweights: IntVector | None = ...,
    aweights: npt.ArrayLike | None = ...,
    sort: SortKind | None = ...,
    cache: bool = ...,
) -> T:
    ...


@overload
def l_moment(
    a: npt.ArrayLike,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: type[np.float64] = ...,
    fweights: IntVector | None = ...,
    aweights: npt.ArrayLike | None = ...,
    sort: SortKind | None = ...,
    cache: bool = ...,
) -> npt.NDArray[np.float64] | np.float64:
    ...


@overload
def l_moment(
    a: npt.ArrayLike,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: np.dtype[T] | type[T],
    fweights: IntVector | None = ...,
    aweights: npt.ArrayLike | None = ...,
    sort: SortKind | None = ...,
    cache: bool = ...,
) -> npt.NDArray[T] | T:
    ...


@overload
def l_moment(
    a: npt.ArrayLike,
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: type[np.float64] = ...,
    fweights: IntVector | None = ...,
    aweights: npt.ArrayLike | None = ...,
    sort: SortKind | None = ...,
    cache: bool = ...,
) -> npt.NDArray[np.float64]:
    ...


@overload
def l_moment(
    a: npt.ArrayLike,
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: np.dtype[T] | type[T],
    fweights: IntVector | None = ...,
    aweights: npt.ArrayLike | None = ...,
    sort: SortKind | None = ...,
    cache: bool = ...,
) -> npt.NDArray[T]:
    ...


def l_moment(
    a: npt.ArrayLike,
    r: IntVector | AnyInt,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    fweights: IntVector | None = None,
    aweights: npt.ArrayLike | None = None,
    sort: SortKind | None = None,
    cache: bool = False,
) -> npt.NDArray[T] | T:
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

            - $(0, 0)$: The original **L**-moment, introduced by Hosking
                in 1990.
            - $(0, t)$: **LL**-moment (**L**inear combination of **L**owest
                order statistics), introduced by Bayazit & Onoz in 2002.
                Assigns more weight to smaller observations.
            - $(s, 0)$: **LH**-moment (**L**inear combination of **H**igher
                order statistics), as described by Wang in 1997.
                Assigns more weight to larger observations.
            - $(t, t)$: **TL**-moment (**T**rimmed L-moment) $\\lambda_r^t$,
                with symmetric trimming. First introduced by Elamir & Seheult
                in 2003, and refined by Hosking in 2007. Generally more robust
                than L-moments. Useful for fitting pathological distributions,
                such as the Cauchy distribution.
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
        sort ('quick' | 'stable' | 'heap'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].
        cache:
            Set to `True` to speed up future L-moment calculations that have
            the same number of observations in `a`, equal `trim`, and equal or
            smaller `r`.

    Returns:
        l:
            The L-moment(s) of the input This is a scalar iff a is 1-d and
            r is a scalar. Otherwise, this is an array with
            `np.ndim(r) + np.ndim(a) - 1` dimensions and shape like
            `(*np.shape(r), *(d for d in np.shape(a) if d != axis))`.

    Examples:
        Calculate the L-location and L-scale from student-T(2) samples, for
        different (symmetric) trim-lengths.

        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_t(2, 99)
        >>> lmo.l_moment(x, [1, 2], trim=(0, 0))
        array([-0.01412282,  0.94063132])
        >>> lmo.l_moment(x, [1, 2], trim=(1/2, 1/2))
        array([-0.02158858,  0.5796519 ])
        >>> lmo.l_moment(x, [1, 2], trim=(1, 1))
        array([-0.0124483 ,  0.40120115])

        The theoretical L-locations are all 0, and the the L-scale are
        `1.1107`, `0.6002` and `0.4165`, respectively.

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

    _r = np.asarray(r)
    r_max = clean_order(np.max(_r))

    # TODO @jorenham: nan handling, see:
    # https://github.com/jorenham/Lmo/issues/70

    # ensure that any inf's (not nan's) are properly trimmed
    s, t = clean_trim(trim)
    if s and isinstance(s, int | np.integer):
        x_k[..., :s] = np.nan_to_num(x_k[..., :s], nan=np.nan)
    if t and isinstance(t, int | np.integer):
        x_k[..., -t:] = np.nan_to_num(x_k[..., -t:], nan=np.nan)

    l_r = np.inner(l_weights(r_max, n, (s, t), cache=cache, dtype=dtype), x_k)

    # we like 0-based indexing; so if P_r starts at r=1, prepend all 1's
    # for r=0 (any zeroth moment is defined to be 1)
    l_r = np.r_[np.ones((1, *l_r.shape[1:]), dtype=l_r.dtype), l_r]

    # l[r] fails when r is e.g. a tuple (valid sequence).
    return cast(npt.NDArray[T] | T, l_r.take(_r, 0))


@overload
def l_ratio(
    a: npt.ArrayLike,
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> np.float64:
    ...


@overload
def l_ratio(
    a: npt.ArrayLike,
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> T:
    ...


@overload
def l_ratio(
    a: npt.ArrayLike,
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> np.float64 | npt.NDArray[np.float64]:
    ...


@overload
def l_ratio(
    a: npt.ArrayLike,
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    ...


@overload
def l_ratio(
    a: npt.ArrayLike,
    r: IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[np.float64]:
    ...


@overload
def l_ratio(
    a: npt.ArrayLike,
    r: IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T]:
    ...


@overload
def l_ratio(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    s: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[np.float64]:
    ...


@overload
def l_ratio(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    s: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int | None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T]:
    ...


def l_ratio(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    r"""
    Estimates the generalized L-moment ratio:

    $$
    \tau^{(s, t)}_{rs} = \frac{
        \lambda^{(s, t)}_r
    }{
        \lambda^{(s, t)}_s
    }
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

        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).lognormal(size=99)
        >>> lmo.l_ratio(x, [1, 2, 3, 4], [0, 0, 2, 2])
        array([1.53196368, 0.77549561, 0.4463163 , 0.29752178])
        >>> lmo.l_ratio(x, [1, 2, 3, 4], [0, 0, 2, 2], trim=(0, 1))
        array([0.75646807, 0.32203446, 0.23887609, 0.07917904])

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
    """  # noqa: D415
    rs = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))
    l_rs = l_moment(a, rs, trim, axis=axis, dtype=dtype, **kwargs)

    return moments_to_ratio(rs, l_rs)


def l_stats(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = (0, 0),
    num: int = 4,
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T]:
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
    return l_ratio(a, r, s, trim=trim, axis=axis, dtype=dtype, **kwargs)


@overload
def l_loc(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> np.float64:
    ...


@overload
def l_loc(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> T:
    ...


@overload
def l_loc(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[np.float64] | np.float64:
    ...


@overload
def l_loc(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    ...


def l_loc(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    r"""
    *L-location* (or *L-loc*): unbiased estimator of the first L-moment,
    $\lambda^{(s, t)}_1$.

    Alias for [`lmo.l_moment(a, 1, *, **)`][lmo.l_moment].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_cauchy(99)
        >>> x.mean()
        -7.5648...
        >>> lmo.l_loc(x)  # no trim; equivalent to the (arithmetic) mean
        -7.5648...
        >>> lmo.l_loc(x, trim=(1, 1))  # TL-location
        -0.15924...
        >>> lmo.l_loc(x, trim=(3/2, 3/2))  # Fractional trimming (only in Lmo)
        -0.085845...


    Notes:
        If `trim = (0, 0)` (default), the L-location is equivalent to the
        [arithmetic mean](https://wikipedia.org/wiki/Arithmetic_mean).

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
        - [`numpy.average`][numpy.average]
    """
    return l_moment(a, 1, trim=trim, axis=axis, dtype=dtype, **kwargs)


@overload
def l_scale(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> np.float64:
    ...


@overload
def l_scale(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> T:
    ...


@overload
def l_scale(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[np.float64] | np.float64:
    ...


@overload
def l_scale(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    ...


def l_scale(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    r"""
    *L-scale*: unbiased estimator of the second L-moment,
    $\lambda^{(s, t)}_2$.

    Alias for [`lmo.l_moment(a, 2, *, **)`][lmo.l_moment].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_cauchy(99)
        >>> x.std()
        72.87715244...
        >>> lmo.l_scale(x)
        9.501123995...
        >>> lmo.l_scale(x, trim=(1, 1))
        0.658993279...

    Notes:
        If `trim = (0, 0)` (default), the L-scale is equivalent to half the
        [Gini mean difference (GMD)](
        https://wikipedia.org/wiki/Gini_mean_difference).

    See Also:
        - [`lmo.l_moment`][lmo.l_moment]
        - [`numpy.std`][numpy.std]
    """
    return l_moment(a, 2, trim, axis=axis, dtype=dtype, **kwargs)


@overload
def l_variation(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> np.float64:
    ...


@overload
def l_variation(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> T:
    ...


@overload
def l_variation(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[np.float64] | np.float64:
    ...


@overload
def l_variation(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    ...


def l_variation(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    r"""
    The *coefficient of L-variation* (or *L-CV*) unbiased sample estimator:

    $$
    \tau^{(s, t)} = \frac{
        \lambda^{(s, t)}_2
    }{
        \lambda^{(s, t)}_1
    }
    $$

    Alias for [`lmo.l_ratio(a, 2, 1, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).pareto(4.2, 99)
        >>> x.std() / x.mean()
        1.32161112...
        >>> lmo.l_variation(x)
        0.59073639...
        >>> lmo.l_variation(x, trim=(0, 1))
        0.55395044...

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
    return l_ratio(a, 2, 1, trim, axis=axis, dtype=dtype, **kwargs)


@overload
def l_skew(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> np.float64:
    ...


@overload
def l_skew(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> T:
    ...


@overload
def l_skew(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[np.float64] | np.float64:
    ...


@overload
def l_skew(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    ...


def l_skew(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    r"""
    Unbiased sample estimator of the *coefficient of L-skewness*, or *L-skew*
    for short:

    $$
    \tau^{(s, t)}_3
        = \frac{
            \lambda^{(s, t)}_3
        }{
            \lambda^{(s, t)}_2
        }
    $$

    Alias for [`lmo.l_ratio(a, 3, 2, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_exponential(99)
        >>> lmo.l_skew(x)
        0.38524343...
        >>> lmo.l_skew(x, trim=(0, 1))
        0.27116139...

    See Also:
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.skew`][scipy.stats.skew]
    """  # noqa: D415
    return l_ratio(a, 3, 2, trim, axis=axis, dtype=dtype, **kwargs)


@overload
def l_kurtosis(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> np.float64:
    ...


@overload
def l_kurtosis(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: None = ...,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> T:
    ...


@overload
def l_kurtosis(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: type[np.float64] = ...,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[np.float64] | np.float64:
    ...


@overload
def l_kurtosis(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = ...,
    *,
    axis: int,
    dtype: np.dtype[T] | type[T],
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    ...


def l_kurtosis(
    a: npt.ArrayLike,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T] | T:
    r"""
    L-kurtosis coefficient; the 4th sample L-moment ratio.

    $$
    \tau^{(s, t)}_4
        = \frac{
            \lambda^{(s, t)}_4
        }{
            \lambda^{(s, t)}_2
        }
    $$

    Alias for [`lmo.l_ratio(a, 4, 2, *, **)`][lmo.l_ratio].

    Examples:
        >>> import lmo, numpy as np
        >>> x = np.random.default_rng(12345).standard_t(2, 99)
        >>> lmo.l_kurtosis(x)
        0.28912787...
        >>> lmo.l_kurtosis(x, trim=(1, 1))
        0.19928182...

    Notes:
        The L-kurtosis $\tau_4$ lies within the interval
        $[-\frac{1}{4}, 1)$, and by the L-skewness $\\tau_3$ as
        $5 \tau_3^2 - 1 \le 4 \tau_4$.

    See Also:
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.kurtosis`][scipy.stats.kurtosis]
    """
    return l_ratio(a, 4, 2, trim, axis=axis, dtype=dtype, **kwargs)


def l_moment_cov(
    a: npt.ArrayLike,
    r_max: SupportsIndex,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T]:
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
    s_b = pwm_beta.cov(a, ks, axis=axis, dtype=dtype, **kwargs)

    # tasty, eh?
    return sandwich(p_l, s_b, dtype=dtype)


def l_ratio_se(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T]:
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
    l_rs = l_moment(a, _rs, trim, axis=axis, dtype=dtype, **kwargs)
    l_r, l_s = l_rs[0], l_rs[1]

    # L-moment auto-covariance matrix
    k_l = l_moment_cov(a, r_max, trim, axis=axis, dtype=dtype, **kwargs)
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
    a: npt.ArrayLike,
    /,
    num: int = 4,
    trim: AnyTrim = (0, 0),
    *,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float64,
    **kwargs: Unpack[LMomentOptions],
) -> npt.NDArray[T]:
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
    return l_ratio_se(a, r, s, trim=trim, axis=axis, dtype=dtype, **kwargs)


def l_moment_influence(
    a: npt.ArrayLike,
    r: SupportsIndex,
    /,
    trim: AnyTrim = (0, 0),
    *,
    sort: SortKind | None = None,
    tol: float = 1e-8,
) -> Callable[[V], V]:
    r"""
    Empirical Influence Function (EIF) of a sample L-moment.

    Notes:
        This function is not vectorized.

    Args:
        a: 1-D array-like containing observed samples.
        r: L-moment order. Must be a non-negative integer.
        trim:
            Left- and right- trim. Can be scalar or 2-tuple of
            non-negative int or float.

    Other parameters:
        sort ('quick' | 'stable' | 'heap'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The (vectorized) empirical influence function.

    """
    _r = clean_order(r)
    s, t = clean_trim(trim)

    x_k = np.sort(a, kind=sort)
    n = len(x_k)

    w_k = l_weights(_r, n, (s, t))[-1]
    l_r = np.inner(w_k, x_k)

    def influence_function(x: V, /) -> V:
        _x = np.asarray(x)

        # ECDF
        # k = np.maximum(np.searchsorted(x_k, _x, side='right') - 1, 0)
        w = cast(V, np.interp(
            _x,
            x_k,
            w_k,
            left=0 if s else w_k[0],
            right=0 if t else w_k[-1],
        ))

        alpha = n * w * np.where(w, _x, 0)
        return cast(V, round0(alpha - l_r, tol=tol)[()])

    influence_function.__doc__ = (
        f'Empirical L-moment influence function for {r=}, {trim=}, and {n=}.'
    )
    # piggyback the L-moment, to avoid recomputing it in l_ratio_influence
    influence_function.l = l_r  # type: ignore
    return influence_function


def l_ratio_influence(
    a: npt.ArrayLike,
    r: SupportsIndex,
    k: SupportsIndex = 2,
    /,
    trim: AnyTrim = (0, 0),
    *,
    sort: SortKind | None = None,
    tol: float = 1e-8,
) -> Callable[[V], V]:
    r"""
    Empirical Influence Function (EIF) of a sample L-moment ratio.

    Notes:
        This function is not vectorized.

    Args:
        a: 1-D array-like containing observed samples.
        r: L-moment ratio order. Must be a non-negative integer.
        k: Denominator L-moment order, defaults to 2.
        trim:
            Left- and right- trim. Can be scalar or 2-tuple of
            non-negative int or float.

    Other parameters:
        sort ('quick' | 'stable' | 'heap'):
            Sorting algorithm, see [`numpy.sort`][numpy.sort].
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The (vectorized) empirical influence function.

    """
    _x = np.sort(a, kind=sort)  # type: ignore
    _r, _k = clean_order(r), clean_order(k)
    n = len(_x)

    eif_r = l_moment_influence(_x, _r, trim, sort='stable', tol=0)
    eif_k = l_moment_influence(_x, _k, trim, sort='stable', tol=0)

    l_r, l_k = cast(tuple[float, float], (eif_r.l, eif_k.l))  # type: ignore
    if abs(l_k) <= tol * abs(l_r):
        msg = f'L-ratio ({r=}, {k=}) denominator is approximately zero.'
        raise ZeroDivisionError(msg)

    t_r = l_r / l_k

    def influence_function(x: V, /) -> V:
        psi_r = eif_r(x)
        # cheat a bit to avoid `inf - inf = nan` situations
        psi_k = np.where(np.isinf(psi_r), 0, eif_k(x))

        return cast(V, round0((psi_r - t_r * psi_k) / l_k, tol=tol)[()])

    influence_function.__doc__ = (
        f'Theoretical influence function for L-moment ratio with r={_r}, '
        f'k={_k}, {trim=}, and {n=}'
    )
    return influence_function
