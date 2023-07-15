"""
Unbiased sample estimators of the generalized trimmed L-moments.
"""

__all__ = (
    'l_weights',
    'l_moment',
    'l_moment_cov',
    'l_ratio',
    'l_ratio_se',
    'l_loc',
    'l_scale',
    'l_variation',
    'l_skew',
    'l_kurtosis',

    'l_moment_from_cdf',
    'l_moment_from_ppf',
)

import functools
import math
from typing import Any, Callable, Final, TypeVar, cast
import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import betainc  # type: ignore [reportUnknownVariableType]

from ._pwm import b_moment_cov, b_weights
from ._utils import clean_order, ensure_axis_at
from .linalg import trim_matrix, sandwich, sh_legendre, sh_jacobi
from .stats import ordered
from .typing import AnyFloat, AnyInt, IntVector, SortKind

T = TypeVar('T', bound=np.floating[Any])


# Low-level weight methods

_L_WEIGHTS_CACHE: Final[
    dict[
        tuple[int, int, int],  # (n, t_1, t_2)
        npt.NDArray[np.floating[Any]]
    ]
] = {}

def _l0_weights(
    r: int,
    n: int,
    /,
    dtype: np.dtype[T] | type[T] = np.float_,
    *,
    enforce_symmetry: bool = True,
) -> npt.NDArray[T]:
    """
    Efficiently calculates the projection matrix $P = [p_{k, i}]_{r \\times n}$
    for the order statistics $x_{i:n}$.
    This way, the $1, 2, ..., r$-th order sample L-moments of some sample vector
    $x$, can be estimated with `np.sort(x) @ l_weights(len(x), r)`.

    Parameters:
        r: The amount of orders to evaluate, i.e. $k = 1, \\dots, r$.
        n: Sample count.
        dtype: Desired output floating data type.

    Other parameters:
        enforce_symmetry:
            If set to False, disables symmetry-based numerical noise correction.

    Returns:
        P_r: 2-D array of shape `(r, n)`.

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    """
    P_r = np.empty((r, n), dtype)

    if r == 0:
        return P_r

    np.matmul(sh_legendre(r), b_weights(r, n, dtype), out=P_r)

    if enforce_symmetry:
        # enforce rotational symmetry of even orders `r = 2, 4, ...`, naturally
        # centering them around 0
        for k in range(2, r + 1, 2):
            p_k: npt.NDArray[T] = P_r[k - 1]

            med = 0.0
            pk_neg, pk_pos = p_k < med, p_k > med
            # n_neg, n_pos = pk_neg.sum(), pk_pos.sum()
            n_neg, n_pos = np.count_nonzero(pk_neg), np.count_nonzero(pk_pos)

            # attempt to correct 1-off asymmetry
            if abs(n_neg - n_pos) == 1:
                if n % 2:
                    # balance the #negative and #positive for odd `n` by
                    # ignoring the center
                    mid = (n - 1) // 2
                    pk_neg[mid] = pk_pos[mid] = False
                    # n_neg, n_pos = pk_neg.sum(), pk_pos.sum()
                    n_neg = np.count_nonzero(pk_neg)
                    n_pos = np.count_nonzero(pk_pos)
                else:
                    # if one side is half of n, set the other to it's negation
                    mid = n // 2
                    if n_neg == mid:
                        pk_pos = ~pk_neg
                        n_pos = n_neg
                    elif n_pos == mid:
                        pk_neg = ~pk_pos
                        n_neg = n_pos

            # attempt to correct any large asymmetry offsets
            # and don't worry; median is O(n)
            if abs(n_neg - n_pos) > 1 and (med := np.median(p_k)):
                pk_neg, pk_pos = p_k < med, p_k > med
                n_neg = np.count_nonzero(pk_neg)
                n_pos = np.count_nonzero(pk_pos)

            if n_neg == n_pos:
                # it's pretty rare that this isn't the case
                p_k[pk_neg] = -p_k[pk_pos][::-1]

        # enforce horizontal (axis 1) symmetry for the odd orders (except k=1)
        # and shift to zero mean
        P_r[2::2, :n // 2] = P_r[2::2, :(n - 1) // 2: -1]
        P_r[2::2] -= P_r[2::2].mean(1, keepdims=True)

    return P_r


def l_weights(
    r: int,
    n: int,
    /,
    trim: tuple[int, int] = (0, 0),
    dtype: np.dtype[T] | type[T] = np.float_,
    *,
    cache: bool = False
) -> npt.NDArray[T]:
    """
    Projection matrix of the first $r$ (T)L-moments for $n$ samples.

    The matrix is a linear combination of the Power Weighted Moment
    (PWM) weight matrix (the sample estimator of $beta_{r_1}$), and the
    shifted Legendre polynomials.

    If `trim != (0, 1)`, a linearized (and corrected) adaptation of the
    recurrence relations from *Hosking (2007)* are applied, as well.

    $$
    (2k + t_1 + t_2 - 1) \\lambda^{(t_1, t_2)}_k
        = (k + t_1 + t_2) \\lambda^{(t_1 - 1, t_2)}_k
        + \\frac{1}{k} (k + 1) (k + t_2) \\lambda^{(t_1 - 1, t_2)}_{k+1}
    $$

    for $t_1 > 0$, and

    $$
    (2k + t_1 + t_2 - 1) \\lambda^{(t_1, t_2)}_k
        = (k + t_1 + t_2) \\lambda^{(t_1, t_2 - 1)}_k
        - \\frac{1}{k} (k + 1) (k + t_1) \\lambda^{(t_1, t_2 - 1)}_{k+1}
    $$

    for $t_2 > 0$.

    TLDR:
        This matrix (linearly) transforms $x_{i:n}$ (i.e. the sorted
        observation vector(s) of size $n$), into (an unbiased estimate of) the
        *generalized trimmed L-moments*, with orders $\\le r$.

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
    cache_key = n, *trim
    if (
        cache_key in _L_WEIGHTS_CACHE
        and (P_r := _L_WEIGHTS_CACHE[cache_key]).shape[0] <= r
    ):
        if P_r.dtype is not np.dtype(dtype):
            P_r = P_r.view(dtype)
        if P_r.shape[0] < r:
            P_r = P_r[:r]

        # ignore if r is larger that what's cached
        if P_r.shape[0] == r:
            assert P_r.shape == (r, n)
            return cast(npt.NDArray[T], P_r)


    if sum(trim) == 0:
        return _l0_weights(r, n, dtype)

    P_r = np.empty((r, n), dtype)

    if r == 0:
        return P_r

    # the k-th TL-(t_1, t_2) weights are a linear combination of L-weights
    # with orders k, ..., k + t_1 + t_2

    np.matmul(
        trim_matrix(r, trim),
        _l0_weights(r + sum(trim), n),
        out=P_r
    )

    # remove numerical noise from the trimmings, and correct for potential
    # shifts in means
    t1, t2 = trim
    P_r[:, :t1] = P_r[:, n - t2:] = 0
    P_r[1:, t1:n - t2] -= P_r[1:, t1:n - t2].mean(1, keepdims=True)

    if cache:
        # memoize, and mark as readonly to avoid corruping the cache
        P_r.setflags(write=False)
        _L_WEIGHTS_CACHE[cache_key] = P_r

    return P_r


# Summary statistics

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
    cache: bool = False,
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
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    """
    # weight-adjusted $x_{i:n}$
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

    r_max = clean_order(np.max(np.asarray(r)))

    # projection matrix
    P_r = l_weights(r_max, n, trim, dtype=dtype, cache=cache)

    l_r = np.inner(P_r, x_k)

    # we like 0-based indexing; so if P_r starts at r=1, prepend all 1's
    # for r=0 (any zeroth moment is defined to be 1)
    l_r = np.r_[np.ones((1, *l_r.shape[1:]), dtype=l_r.dtype), l_r]

    assert np.all(l_r[0] == 1)
    assert len(l_r) == r_max + 1, (l_r.shape, r_max)

    # l[r] fails when r is e.g. a tuple (valid sequence).
    return l_r.take(r, 0)


def l_moment_cov(
    a: npt.ArrayLike,
    r_max: AnyInt,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
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
        errors, by taking the square root of the variances (the diagnoal of the
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

    """
    ks = int(r_max + sum(trim))
    if ks < r_max:
        raise ValueError('trimmings must be positive')

    # PWM covariance matrix
    S_b = b_moment_cov(a, ks, axis=axis, dtype=dtype, **kwargs)

    # projection matrix: PWMs -> generalized trimmed L-moments
    P_l: npt.NDArray[np.floating[Any]]
    P_l = trim_matrix(int(r_max), trim=trim, dtype=dtype) @ sh_legendre(ks)
    # clean some numerical noise
    P_l = np.round(P_l, 12) + 0.  # pyright: ignore [reportUnknownMemberType]

    # tasty, eh?
    return sandwich(P_l, S_b, dtype=dtype)


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
    if r_eq_s.ndim < l_rs.ndim - 1:
        r_eq_s = r_eq_s.reshape(
            r_eq_s.shape + (1,) * (l_rs.ndim - r_eq_s.ndim - 1)
        )

    with np.errstate(divide='ignore'):
        return np.where(
            r_eq_s,
            np.ones_like(l_rs[0]),
            np.divide(l_rs[0], l_rs[1], where=~r_eq_s)
        )[()]



def l_ratio_se(
    a: npt.ArrayLike,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: tuple[int, int] = (0, 0),
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
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
    r_max: AnyInt = np.amax(  # pyright: ignore [reportUnknownMemberType]
        np.r_[_r, _s].ravel()
    )

    # L-moments
    l_rs = cast(npt.NDArray[T], l_moment(a, _rs, trim, axis, dtype, **kwargs))
    l_r, l_s = l_rs[0], l_rs[1]

    # L-moment auto-covariance matrix
    S_l = l_moment_cov(a, r_max, trim, axis, dtype, **kwargs)
    # prepend the "zeroth" moment, with has 0 (co)variance
    S_l = np.pad(S_l, (1, 0), constant_values=0)

    s_rr = S_l[_r, _r]  # Var[l_r]
    s_ss = S_l[_s, _s]  # Var[l_r]
    s_rs = S_l[_r, _s]  # Cov[l_r, l_s]

    # the classic approximation to propagation of uncertainty for an RV ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        # TODO: np.piecewiese ?
        _s_tt = (l_r / l_s)**2 * (
            s_rr / l_r**2 +
            s_ss / l_s**2 -
            2 * s_rs / (l_r * l_s)
        )
        # Var[l_r / l_r] = Var[1] = 0
        _s_tt = np.where(_s == 0, s_rr, _s_tt)
        # Var[l_r / l_0] == Var[l_r / 1] == Var[l_r]
        s_tt = np.where(_r == _s, 0, _s_tt)

    return np.sqrt(s_tt)


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
        -7.56485034...
        >>> lmo.l_loc(x)
        -7.56485034...
        >>> lmo.l_loc(x, trim=(1, 1))
        -0.15924180...

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
        0.38524343...
        >>> lmo.l_skew(x, trim=(0, 1))
        0.27116139...

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
        0.28912787...
        >>> lmo.l_kurtosis(x, trim=(1, 1))
        0.19928182...

    Notes:
        The L-kurtosis $\\tau_4$ lies within the interval
        $[-\\frac{1}{4}, 1)$, and by the L-skewness $\\tau_3$ as
        $5 \\tau_3^2 - 1 \\le 4 \\tau_4$.

    See Also:
        - [`lmo.l_ratio`][lmo.l_ratio]
        - [`scipy.stats.kurtosis`][scipy.stats.kurtosis]

    """
    return l_ratio(a, 4, 2, trim, axis, dtype, **kwargs)


# Numerical calculation of population L-moments


def l_moment_from_cdf(
    cdf: Callable[[float], float],
    r: AnyInt | IntVector,
    /,
    trim: tuple[AnyFloat, AnyFloat] = (0, 0),
    support: tuple[AnyFloat, AnyFloat] = (-np.inf, np.inf),
) -> float | npt.NDArray[np.float_]:
    # TODO: docstring

    _r = np.asanyarray(r)
    if not np.issubdtype(_r.dtype, np.integer):
        raise TypeError(f'r must be integer-valued, got {_r.dtype.str!r}')
    if _r.size == 0:
        raise ValueError('no r provided')
    if np.any(_r < 0):
        raise ValueError('r must be non-negative')

    s, t = np.asanyarray(trim)

    r_max = clean_order(np.max(_r))
    r_vals, r_idxs = np.unique(_r, return_inverse=True)
    assert r_vals.ndim == 1

    # caching F(x) function only makes sense for multiple quad calls
    F = functools.cache(cdf) if np.count_nonzero(r_vals) > 1 else cdf

    # shifted Jacobi polynomial coefficients
    j = sh_jacobi(r_max - 1, t + 1, s + 1)

    # lazy import (don't worry; python imports are cached)
    from scipy.integrate import quad, IntegrationWarning  # type: ignore

    l_r = np.empty(r_vals.shape)
    for i, r_val in np.ndenumerate(r_vals):
        if r_val == 0:
            # zeroth l-moment is always 1
            l_r[i] = 1
            continue

        if r_val == 1:
            if s == t == 0:
                def integrand(x: float, *args: Any) -> float:
                    return (x >= 0) - F(x, *args)
            else:
                def integrand(x: float, *args: Any) -> float:
                    p = F(x, *args)
                    return (x >= 0) - betainc(s + 1, t + 1, p)  # type: ignore

            const = 1

        else:
            # prepare the powers to use for evaluating the polynomial
            k = cast(npt.NDArray[np.int_], np.arange(r_val - 1))
            # grab the non-zero jacobi polynomial coefficients for k=r-1
            j_k = j[r_val - 2, :r_val - 1]

            def integrand(x: float, *args: Any) -> float:
                # evaluate the jacobi polynomial for p at r-1 with (t, s)
                # and multiply by the weight function
                p = F(x, *args)
                return p ** (s + 1) * (1 - p) ** (t + 1) * (j_k @ p**k)

            const = np.exp(
                math.lgamma(r_val - 1)
                + math.lgamma(r_val + s + t + 1)
                - math.lgamma(r_val + s)
                - math.lgamma(r_val + t)
            ) / r_val

        # numerical integration
        quad_val, _, _, *quad_tail = cast(
            tuple[float, float, dict[str, Any]]
            | tuple[float, float, dict[str, Any], str],
            quad(integrand, *support, full_output=True)
        )

        if quad_tail:
            quad_msg = quad_tail[0]
            warnings.warn(
                f"'scipy.integrate.quad' failed: \n{quad_msg}",
                cast(type[UserWarning], IntegrationWarning),
                stacklevel=2
            )
            l_r[i] = np.nan
            continue

        l_r[i] = const * quad_val

    return np.round(l_r, 12)[r_idxs if _r.ndim > 0 else 0] + .0


def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: AnyInt | IntVector,
    /,
    trim: tuple[AnyFloat, AnyFloat] = (0, 0),
    support: tuple[AnyFloat, AnyFloat] = (0, 1),
) -> float | npt.NDArray[np.float_]:
    # TODO: docstring

    _r = np.asanyarray(r)
    if not np.issubdtype(_r.dtype, np.integer):
        raise TypeError(f'r must be integer-valued, got {_r.dtype.str!r}')
    if _r.size == 0:
        raise ValueError('no r provided')
    if np.any(_r < 0):
        raise ValueError('r must be non-negative')

    s, t = np.asanyarray(trim)

    r_max = clean_order(np.max(_r))
    r_vals, r_idxs = np.unique(_r, return_inverse=True)
    assert r_vals.ndim == 1

    def _w(p: float, *args: Any) -> float:
        return p**s * (1 - p)**t * ppf(p, *args)

    # caching the weight function only makes sense for multiple quad calls
    w = functools.cache(_w) if len(r_vals) > 1 else _w

    # shifted Jacobi polynomial coefficients
    j = sh_jacobi(r_max, t, s)

    # lazy import (don't worry; python imports are cached)
    from scipy.integrate import quad, IntegrationWarning  # type: ignore

    l_r = np.empty(r_vals.shape)
    for i, r_val in np.ndenumerate(r_vals):
        if r_val == 0:
            # zeroth l-moment is always 1
            l_r[i] = 1
            continue

        # prepare the powers to use for evaluating the polynomial
        k = cast(npt.NDArray[np.int_], np.arange(r_val))
        # grab the non-zero jacobi polynomial coefficients for k=r-1
        j_k = j[r_val - 1, :r_val]

        def integrand(p: float) -> float:
            # evaluate the jacobi polynomial for p at r-1 with (t, s)
            # and multiply by the weight function
            return w(p) * (j_k @ p**k)  # type: ignore

        # numerical integration
        quad_val, _, _, *quad_tail = cast(
            tuple[float, float, dict[str, Any]]
            | tuple[float, float, dict[str, Any], str],
            quad(integrand, *support, full_output=True)
        )
        if quad_tail:
            quad_msg = quad_tail[0]
            warnings.warn(
                f"'scipy.integrate.quad' failed: \n{quad_msg}",
                cast(type[UserWarning], IntegrationWarning),
                stacklevel=2
            )
            l_r[i] = np.nan
            continue

        # constant combinatorial factor
        const = np.exp(
            math.lgamma(r_val)
            + math.lgamma(r_val + s + t + 1)
            - math.lgamma(r_val + s)
            - math.lgamma(r_val + t)
        ) / r_val

        l_r[i] = const * quad_val

    return np.round(l_r, 12)[r_idxs if _r.ndim > 0 else 0] + .0
