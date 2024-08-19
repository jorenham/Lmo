"""Hypothesis tests, estimator properties, and performance metrics."""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from math import lgamma
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    NamedTuple,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad  # pyright: ignore[reportUnknownVariableType]
from scipy.optimize import (
    OptimizeWarning,
    minimize,  # pyright: ignore[reportUnknownVariableType]
)
from scipy.special import chdtrc
from scipy.stats.distributions import rv_continuous, rv_frozen

import lmo.typing.np as lnpt
import lmo.typing.scipy as lspt
from . import constants
from ._lm import l_ratio
from ._poly import extrema_jacobi
from ._utils import clean_orders, clean_trim
from .special import fpow
from .typing import AnyOrder, AnyOrderND, AnyTrim


if TYPE_CHECKING:
    from .contrib.scipy_stats import l_rv_generic


__all__ = (
    'normaltest',
    'l_moment_gof',
    'l_stats_gof',

    'l_moment_bounds',
    'l_ratio_bounds',

    'rejection_point',
    'error_sensitivity',
    'shift_sensitivity',
)


_T = TypeVar('_T')

_Tuple2: TypeAlias = tuple[_T, _T]
_ArrF8: TypeAlias = npt.NDArray[np.float64]

_MIN_RHO: Final[float] = 1e-5


class HypothesisTestResult(NamedTuple):
    r"""
    Results of a hypothesis test.

    Attributes:
        statistic:
            The raw test statistic. Its distribution depends on the specific
            test implementation.
        pvalue:
            Two-sided probability value corresponding to the the null
            hypothesis, $H_0$.
    """

    statistic: float | _ArrF8
    pvalue: float | _ArrF8

    @property
    def is_valid(self) -> np.bool_ | npt.NDArray[np.bool_]:
        """Check if the statistic is finite and not `nan`."""
        return np.isfinite(self.statistic)

    def is_significant(
        self,
        level: float | np.floating[Any] = 0.05,
        /,
    ) -> np.bool_ | npt.NDArray[np.bool_]:
        """
        Whether or not the null hypothesis can be rejected, with a certain
        confidence level (5% by default).
        """
        if not (0 < level < 1):
            msg = 'significance level must lie between 0 and 1'
            raise ValueError(msg)
        return self.pvalue < np.float64(level)


def normaltest(
    a: lnpt.AnyArrayFloat,
    /,
    *,
    axis: int | None = None,
) -> HypothesisTestResult:
    r"""
    Statistical hypothesis test for **non**-normality, using the L-skewness
    and L-kurtosis coefficients on the sample data..

    Adapted from Harri & Coble (2011), and includes Hosking's correction.

    Definition:
        - H0: The data was drawn from a normal distribution.
        - H1: The data was drawn from a non-normal distribution.

    Examples:
        Compare the testing power with
        [`scipy.stats.normaltest`][scipy.stats.normaltest] given 10.000 samples
        from a contaminated normal distribution.

        >>> import numpy as np
        >>> from lmo.diagnostic import normaltest
        >>> from scipy.stats import normaltest as normaltest_scipy
        >>> rng = np.random.default_rng(12345)
        >>> n = 10_000
        >>> x = 0.9 * rng.normal(0, 1, n) + 0.1 * rng.normal(0, 9, n)
        >>> normaltest(x)[1]
        0.04806618
        >>> normaltest_scipy(x)[1]
        0.08435627

        At a 5% significance level, Lmo's test is significant (i.e. indicating
        non-normality), whereas scipy's test isn't (i.e. inconclusive).

    Args:
        a: Array-like of sample data.
        axis: Axis along which to compute the test.

    Returns:
        A named tuple with:

            - `statistic`: The $\tau^2_{3, 4}$ test statistic.
            - `pvalue`: Two-sided chi squared probability for $H_0$.

    References:
        [A. Harri & K.H. Coble (2011) - Normality testing: Two new tests
        using L-moments](https://doi.org/10.1080/02664763.2010.498508)
    """
    x = np.asanyarray(a)

    # sample size
    n = x.size if axis is None else x.shape[axis]

    # L-skew and L-kurtosis
    t3, t4 = l_ratio(a, [3, 4], 2, axis=axis)

    # theoretical L-skew and L-kurtosis of the normal distribution (for all
    # loc/mu and scale/sigma)
    tau3, tau4 = .0, 60 * constants.theta_m_bar - 9

    z3 = (t3 - tau3) / np.sqrt(
        0.1866 / n + (np.sqrt(0.8000) / n) ** 2,
    )
    z4 = (t4 - tau4) / np.sqrt(
        0.0883 / n + (np.sqrt(0.6800) / n) ** 2 + (np.cbrt(4.9000) / n) ** 3,
    )

    k2 = z3**2 + z4**2

    # special case of the chi^2 survival function for k=2 degrees of freedom
    p_value = np.exp(-k2 / 2)

    return HypothesisTestResult(k2, p_value)


def _gof_stat_single(l_obs: _ArrF8, l_exp: _ArrF8, cov: _ArrF8) -> np.float64:
    err = l_obs - l_exp
    prec = np.linalg.inv(cov)  # precision matrix
    return cast(np.float64, err.T @ prec @ err)


_gof_stat = cast(
    Callable[[_ArrF8, _ArrF8, _ArrF8], _ArrF8],
    np.vectorize(
        _gof_stat_single,
        otypes=[float],
        excluded={1, 2},
        signature='(n)->()',
    ),
)


def l_moment_gof(
    rv_or_cdf: lspt.AnyRV | Callable[[float], float],
    l_moments: _ArrF8,
    n_obs: int,
    /,
    trim: AnyTrim = 0,
    **kwargs: Any,
) -> HypothesisTestResult:
    r"""
    Goodness-of-fit (GOF) hypothesis test for the null hypothesis that the
    observed L-moments come from a distribution with the given
    [`scipy.stats`][scipy.stats] distribution or cumulative distribution
    function (CDF).

    - `H0`: The theoretical probability distribution, with the given CDF,
        is a good fit for the observed L-moments.
    - `H1`: The distribution is not a good fit for the observed L-moments.

    The test statistic is the squared Mahalanobis distance between the $n$
    observed L-moments, and the theoretical L-moments. It asymptically (in
    sample size) follows the
    [$\chi^2$](https://wikipedia.org/wiki/Chi-squared_distribution)
    distribution, with $n$ degrees of freedom.

    The sample L-moments are expected to be of consecutive orders
    $r = 1, 2, \dots, n$.
    Generally, the amount of L-moments $n$ should not be less than the amount
    of parameters of the distribution, including the location and scale
    parameters. Therefore, it is required to have $n \ge 2$.

    Notes:
        The theoretical L-moments and their covariance matrix are calculated
        from the CDF using numerical integration
        ([`scipy.integrate.quad`][scipy.integrate.quad] and
        [`scipy.integrate.nquad`][scipy.integrate.nquad]).
        Undefined or infinite integrals cannot be detected, in which case the
        results might be incorrect.

        If an [`IntegrationWarning`][scipy.integrate.IntegrationWarning] is
        issued, or the function is very slow, then the results are probably
        incorrect, and larger degrees of trimming should be used.

    Examples:
        Test if the samples are drawn from a normal distribution.

        >>> import lmo
        >>> import numpy as np
        >>> from lmo.diagnostic import l_moment_gof
        >>> from scipy.stats import norm
        >>> rng = np.random.default_rng(12345)
        >>> X = norm(13.12, 1.66)
        >>> n = 1_000
        >>> x = X.rvs(n, random_state=rng)
        >>> x_lm = lmo.l_moment(x, [1, 2, 3, 4])
        >>> l_moment_gof(X, x_lm, n).pvalue
        0.82597

        Contaminated samples:

        >>> y = 0.9 * x + 0.1 * rng.normal(X.mean(), X.std() * 10, n)
        >>> y_lm = lmo.l_moment(y, [1, 2, 3, 4])
        >>> y_lm.round(3)
        array([13.193, 1.286, 0.006, 0.168])
        >>> l_moment_gof(X, y_lm, n).pvalue
        0.0


    See Also:
        - [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf]
        - ['l_moment_cov_from_cdf'][lmo.theoretical.l_moment_cov_from_cdf]

    """
    l_r = np.asarray_chkfinite(l_moments)

    if (n := len(l_r)) < 2:
        msg = f'at least the first 2 L-moments are required, got {n}'
        raise TypeError(msg)

    r = np.arange(1, 1 + n)

    if isinstance(rv_or_cdf, rv_continuous.__base__ | rv_frozen):
        rv = cast('l_rv_generic', rv_or_cdf)
        lambda_r = rv.l_moment(r, trim=trim, **kwargs)
        lambda_rr = rv.l_moments_cov(n, trim=trim, **kwargs)
    else:
        from .theoretical import l_moment_cov_from_cdf, l_moment_from_cdf

        cdf = cast(Callable[[float], float], rv_or_cdf)
        lambda_r = l_moment_from_cdf(cdf, r, trim, **kwargs)
        lambda_rr = l_moment_cov_from_cdf(cdf, n, trim, **kwargs)

    stat = n_obs * _gof_stat(l_r.T, lambda_r, lambda_rr).T[()]
    pval = cast(float | _ArrF8, chdtrc(n, stat))
    return HypothesisTestResult(stat, pval)


def l_stats_gof(
    rv_or_cdf: lspt.AnyRV | Callable[[float], float],
    l_stats: _ArrF8,
    n_obs: int,
    /,
    trim: AnyTrim = 0,
    **kwargs: Any,
) -> HypothesisTestResult:
    """
    Analogous to [`lmo.diagnostic.l_moment_gof`][lmo.diagnostic.l_moment_gof],
    but using the L-stats (see [`lmo.l_stats`][lmo.l_stats]).
    """
    t_r = np.asarray_chkfinite(l_stats)

    if (n := t_r.shape[0]) < 2:
        msg = f'at least 2 L-stats are required, got {n}'
        raise TypeError(msg)

    if isinstance(rv_or_cdf, rv_continuous.__base__ | rv_frozen):
        rv = cast('l_rv_generic', rv_or_cdf)
        tau_r = rv.l_stats(moments=n, trim=trim, **kwargs)
        tau_rr = rv.l_stats_cov(moments=n, trim=trim, **kwargs)
    else:
        from .theoretical import l_stats_cov_from_cdf, l_stats_from_cdf

        cdf = cast(Callable[[float], float], rv_or_cdf)
        tau_r = l_stats_from_cdf(cdf, n, trim, **kwargs)
        tau_rr = l_stats_cov_from_cdf(cdf, n, trim, **kwargs)

    stat = n_obs * _gof_stat(t_r.T, tau_r, tau_rr).T[()]
    pval = cast(float | _ArrF8, chdtrc(n, stat))
    return HypothesisTestResult(stat, pval)


def _lm2_bounds_single(r: int, trim: _Tuple2[float]) -> float:
    if r == 1:
        return float('inf')

    match trim:
        case (0, 0):
            return 1 / (2 * r - 1)
        case (0, 1) | (1, 0):
            return (r + 1)**2 / (r * (2 * r - 1) * (2 * r + 1))
        case (1, 1):
            return (
                (r + 1)**2 * (r + 2)**2
                / (2 * r**2 * (2 * r - 1) * (2 * r + 1) * (2 * r + 1))
            )
        case (s, t):
            return np.exp(
                lgamma(r - .5)
                - lgamma(s + t + 1)
                + lgamma(s + .5)
                - lgamma(r + s)
                + lgamma(t + .5)
                - lgamma(r + t)
                + lgamma(r + s + t + 1) * 2
                - lgamma(r + s + t + .5),
            ) / (np.pi * 2 * r**2)


_lm2_bounds = cast(
    Callable[[AnyOrderND, _Tuple2[float]], _ArrF8],
    np.vectorize(
        _lm2_bounds_single,
        otypes=[float],
        excluded={1},
        signature='()->()',
    ),
)


@overload
def l_moment_bounds(
    r: AnyOrderND, /,
    trim: AnyTrim = ...,
    scale: float = ...,
) -> _ArrF8: ...
@overload
def l_moment_bounds(
    r: AnyOrder, /,
    trim: AnyTrim = ...,
    scale: float = ...,
) -> float: ...
def l_moment_bounds(
    r: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = 0,
    scale: float = 1.0,
) -> float | _ArrF8:
    r"""
    Returns the absolute upper bounds $L^{(s,t)}_r$ on L-moments
    $\lambda^{(s,t)}_r$, proportional to the scale $\sigma_X$ (standard
    deviation) of the probability distribution of random variable $X$.
    So $\left| \lambda^{(s,t)}_r(X) \right| \le \sigma_X \, L^{(s,t)}_r$,
    given that standard deviation $\sigma_X$ of $X$ exists and is finite.

    Warning:
        These bounds do not apply to distributions with undefined variance,
        e.g. the Cauchy distribution, even if trimmed L-moments are used.
        Distributions with infinite variance (e.g. Student's t with $\nu=2$)
        are a grey area:

        For the L-scale ($r=2$), the corresponding bound will not be a valid
        one. However, it can still be used to find the L-ratio bounds, because
        the $\sigma_X$ terms will cancel out.
        Doing this is not for the faint of heart, as it requires dividing
        infinity by infinity. So be sure to wear safety glasses.

    The bounds are derived by applying the [Cauchy-Schwarz inequality](
        https://wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality) to the
    covariance-based definition of generalized trimmed L-moment, for $r > 1$:

    $$
    \lambda^{(s,t)}_r(X) =
        \frac{r+s+t}{r}
        \frac{B(r,\, r+s+t)}{B(r+s,\, r+t)}
        \mathrm{Cov}\left[
            X,\;
            F(X)^s
            \big(1 - F(X)\big)^t
            P^{(\alpha, \beta)}_r(X)
        \right]
    \;,
    $$

    where $B$ is the
    [Beta function](https://mathworld.wolfram.com/BetaFunction.html),
    $P^{(\alpha, \beta)}_r$ the
    [Jacobi polynomial](https://mathworld.wolfram.com/JacobiPolynomial.html),
    and $F$ the cumulative distribution function of random variable $X$.

    After a lot of work, one can (and one did) derive the closed-form
    inequality:

    $$
    \left| \lambda^{(s,t)}_r(X) \right| \le
        \frac{\sigma_X}{\sqrt{2 \pi}}
        \frac{\Gamma(r+s+t+1)}{r}
        \sqrt{\frac{
            B(r-\frac{1}{2}, s+\frac{1}{2}, t+\frac{1}{2})
        }{
            \Gamma(s+t+1) \Gamma(r+s) \Gamma(r+t)
        }}
    $$

    for $r \in \mathbb{N}_{\ge 2}$ and $s, t \in \mathbb{R}_{\ge 0}$, where
    $\Gamma$ is the
    [Gamma function](https://mathworld.wolfram.com/GammaFunction.html),
    and $B$ the multivariate Beta function

    For the untrimmed L-moments, this simplifies to

    $$
    \left| \lambda_r(X) \right| \le \frac{\sigma_X}{\sqrt{2 r - 1}} \,.
    $$

    Notes:
        For $r=1$ there are no bounds, i.e. `float('inf')` is returned.

        There are no references; this novel finding is not (yet..?) published
        by the author, [@jorenham](https://github.com/jorenham/).

    Args:
        r:
            The L-moment order(s), non-negative integer or array-like of
            integers.
        trim:
            Left- and right-trim orders $(s, t)$, as a tuple of non-negative
            ints or floats.
        scale:
            The standard deviation $\sigma_X$ of the random variable $X$.
            Defaults to 1.

    Returns:
        out: float array or scalar like `r`.

    See Also:
        - [`l_ratio_bounds`][lmo.diagnostic.l_ratio_bounds]
        - [`lmo.l_moment`][lmo.l_moment]

    """
    _r = clean_orders(np.asarray(r), rmin=1)
    _trim = clean_trim(trim)
    return scale * np.sqrt(_lm2_bounds(_r, _trim))[()]


@overload
def l_ratio_bounds(
    r: AnyOrderND,
    /,
    trim: AnyTrim = ...,
    *,
    legacy: bool = ...,
) -> _Tuple2[_ArrF8]: ...
@overload
def l_ratio_bounds(
    r: AnyOrder,
    /,
    trim: AnyTrim = ...,
    *,
    legacy: bool = ...,
) -> _Tuple2[float]: ...
def l_ratio_bounds(
    r: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = 0,
    *,
    legacy: bool = False,
) -> _Tuple2[float | _ArrF8]:
    r"""
    Unlike the standardized product-moments, the L-moment ratio's with
    \( r \ge 2 \) are bounded above and below.

    Specifically, Hosking derived in 2007 that

    \[
        | \tlratio{r}{s,t}| \le
            \frac 2 r
            \frac{\ffact{r + s + t}{r - 2}}{\ffact{r - 1 + s \wedge t}{r - 2}}
            .
    \]

    But this derivation relies on unnecessarily loose Jacobi polynomial bounds.
    If the actual min and max of the Jacobi polynomials are used instead,
    the following (tighter) inequality is obtained:

    \[
        \frac{\dot{w}_r^{(s, t)}}{\dot{w}_2^{(s, t)}}
        \min_{u \in [0, 1]} \left[ \shjacobi{r - 1}{t + 1}{s + 1}{u} \right]
        \le
        \tlratio{s, t}{r}
        \le
        \frac{\dot{w}_r^{(s, t)}}{\dot{w}_2^{(s, t)}}
        \max_{0 \le u \le 1} \left[ \shjacobi{r - 1}{t + 1}{s + 1}{u} \right],
    \]

    where

    \[
        \dot{w}_r^{(s, t)} =
            \frac{\B(r - 1,\ r + s + t + 1)}{r \B(r + s,\ r + t)}.
    \]

    Examples:
        Without trim, the lower- and upper-bounds of the L-skewness and
        L-kurtosis are:

        >>> l_ratio_bounds(3)
        (-1.0, 1.0)
        >>> l_ratio_bounds(4)
        (-0.25, 1.0)

        For the L-kurtosis, the "legacy" bounds by Hosking (2007) are clearly
        looser:

        >>> l_ratio_bounds(4, legacy=True)
        (-1.0, 1.0)

        For the symmetrically trimmed TL-moment ratio's:

        >>> l_ratio_bounds(3, trim=3)
        (-1.2, 1.2)
        >>> l_ratio_bounds(4, trim=3)
        (-0.15, 1.5)

        Similarly, those of the LL-ratio's are

        >>> l_ratio_bounds(3, trim=(0, 3))
        (-0.8, 2.0)
        >>> l_ratio_bounds(4, trim=(0, 3))
        (-0.233333, 3.5)

        The LH-skewness bounds are "flipped" w.r.t to the LL-skewness,
        but they are the same for the L*-kurtosis:

        >>> l_ratio_bounds(3, trim=(3, 0))
        (-2.0, 0.8)
        >>> l_ratio_bounds(4, trim=(3, 0))
        (-0.233333, 3.5)

        The bounds of multiple L-ratio's can be calculated in one shot:
        >>> np.stack(l_ratio_bounds([3, 4, 5, 6], trim=(1, 2)))
        array([[-1.        , -0.19444444, -1.12      , -0.14945848],
               [ 1.33333333,  1.75      ,  2.24      ,  2.8       ]])


    Args:
        r: Scalar or array-like with the L-moment ratio order(s).
        trim: L-moment ratio trim-length(s).
        legacy: If set to `True`, will use the (looser) by Hosking (2007).

    Returns:
        A 2-tuple with arrays or scalars, of the lower- and upper bounds.

    See Also:
        - [`l_ratio`][lmo.l_ratio]
        - [`l_ratio_se`][lmo.l_ratio_se]
        - [`diagnostic.l_moment_bounds`][lmo.diagnostic.l_moment_bounds]

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
        L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    """
    _r = clean_orders(np.asarray(r))
    s, t = clean_trim(trim)

    t_min = np.empty(_r.shape)
    t_max = np.empty(_r.shape)

    _cache: dict[int, _Tuple2[float]] = {}
    for i, ri in np.ndenumerate(_r):
        _ri = cast(int, ri)
        if _ri in _cache:
            t_min[i], t_max[i] = _cache[_ri]

        if _ri == 1:
            # L-loc / L-scale; unbounded
            t_min[i], t_max[i] = -np.inf, np.inf
        elif _ri in {0, 2}:  # or s == t == 0:
            t_min[i] = t_max[i] = 1
        elif legacy:
            t_absmax = (
                2
                * fpow(_ri + s + t, _ri - 2)
                / fpow(_ri + min(s, t) - 1, _ri - 2)
                / _ri
            )
            t_min[i] = -t_absmax
            t_max[i] = t_absmax
        else:
            cr_c2 = 2 * (
                np.exp(
                    lgamma(_ri - 1)
                    - np.log(_ri)
                    + lgamma(s + 2)
                    - lgamma(_ri + s)
                    + lgamma(t + 2)
                    - lgamma(_ri + t)
                    + lgamma(_ri + s + t + 1)
                    - lgamma(s + t + 3)  # noqa: COM812
                )
            )

            p_min, p_max = extrema_jacobi(_ri - 2, t + 1, s + 1)
            assert p_min < 0 < p_max, (p_min, p_max)

            t_min[i] = cr_c2 * p_min
            t_max[i] = cr_c2 * p_max

        _cache[_ri] = t_min[i], t_max[i]

    return t_min.round(12)[()], t_max.round(12)[()]


def rejection_point(
    influence_fn: Callable[[float], float],
    /,
    rho_min: float = 0,
    rho_max: float = np.inf,
) -> float:
    r"""
    Evaluate the approximate *rejection point* of an influence function
    $\psi_{T|F}(x)$ given a *statistical functional* $T$ (e.g. an L-moment)
    and cumulative distribution function $F(x)$.

    $$
    \rho^*_{T|F} = \inf_{r>0} \left\{
        r: | \psi_{T|F}(x) | \le \epsilon, \, |x| > r
    \right\} \;
    $$

    with a $\epsilon$ a small positive number, corresponding to the `tol` param
    of e.g. [l_moment_influence
    ][lmo.contrib.scipy_stats.l_rv_generic.l_moment_influence], which defaults
    to `1e-8`.

    Examples:
        The untrimmed L-location isn't robust, e.g. with the standard normal
        distribution:

        >>> import numpy as np
        >>> from scipy.stats import distributions as dists
        >>> from lmo.diagnostic import rejection_point
        >>> if_l_loc_norm = dists.norm.l_moment_influence(1, trim=0)
        >>> if_l_loc_norm(np.inf)
        inf
        >>> rejection_point(if_l_loc_norm)
        nan

        For the TL-location of the Gaussian distribution, and even for the
        Student's t distribution with 4 degrees of freedom (3 also works, but
        is very slow), they exist.

        >>> influence_norm = dists.norm.l_moment_influence(1, trim=1)
        >>> influence_t4 = dists.t(4).l_moment_influence(1, trim=1)
        >>> influence_norm(np.inf), influence_t4(np.inf)
        (0.0, 0.0)
        >>> rejection_point(influence_norm), rejection_point(influence_t4)
        (6.0, 206.0)

    Notes:
        Large rejection points (e.g. >1000) are unlikely to be found.

        For instance, that of the TL-location of the Student's t distribution
        with 2 degrees of freedom lies between somewhere `1e4` and `1e5`, but
        will not be found. In this case, using `trim=2` will return `166.0`.

    Args:
        influence_fn: Univariate influence function.
        rho_min:
            The minimum $\rho^*_{T|F}$ of the search space.
            Must be finite and non-negative.
            Defaults to $0$.
        rho_max:
            The maximum $\rho^*_{T|F}$ of the search space.
            Must be larger than `rho_min`.
            Defaults to $\infty$.

    Returns:
        A finite or infinite scalar.

    See Also:
        - [`lmo.contrib.scipy_stats.l_rv_generic.l_moment_influence`
        ][lmo.contrib.scipy_stats.l_rv_generic.l_moment_influence]
        - [`error_sensitivity`][lmo.diagnostic.error_sensitivity]

    """
    if not 0 <= rho_min < rho_max:
        msg = f'expected 0 <= rho_min < rho_max, got {rho_min=} and {rho_max=}'
        raise ValueError(msg)

    if influence_fn(rho_max) != 0 or influence_fn(-rho_max) != 0:
        return np.nan

    def integrand(x: float) -> float:
        return max(abs(influence_fn(-x)), abs(influence_fn(x)))

    def obj(r: _ArrF8) -> float:
        return quad(integrand, r[0], np.inf)[0]  # pyright: ignore[reportUnknownVariableType]

    res = cast(
        lspt.OptimizeResult,
        minimize(
            obj,
            bounds=[(rho_min, rho_max)],
            x0=[rho_min],
            method='COBYLA',
        ),
    )

    rho = cast(float, res.x[0])
    if rho <= _MIN_RHO or influence_fn(-rho) or influence_fn(rho):
        return np.nan

    return rho


def error_sensitivity(
    influence_fn: Callable[[float], float],
    /,
    domain: _Tuple2[float] = (-math.inf, math.inf),
) -> float:
    r"""
    Evaluate the *gross-error sensitivity* of an influence function
    $\psi_{T|F}(x)$ given a *statistical functional* $T$ (e.g. an L-moment)
    and cumulative distribution function $F(x)$.

    $$
    \gamma^*_{T|F} = \max_{x} \left| \psi_{T|F}(x) \right|
    $$

    Examples:
        Evaluate the gross-error sensitivity of the standard exponential
        distribution's LL-skewness ($\tau^{(0, 1)}_3$) and LL-kurtosis
        ($\tau^{(0, 1)}_4$) coefficients:

        >>> from lmo.diagnostic import error_sensitivity
        >>> from scipy.stats import expon
        >>> ll_skew_if = expon.l_ratio_influence(3, 2, trim=(0, 1))
        >>> ll_kurt_if = expon.l_ratio_influence(4, 2, trim=(0, 1))
        >>> error_sensitivity(ll_skew_if, domain=(0, float('inf')))
        1.814657
        >>> error_sensitivity(ll_kurt_if, domain=(0, float('inf')))
        1.377743

    Args:
        influence_fn: Univariate influence function.
        domain: Domain of the CDF. Defaults to $(-\infty, \infty)$.

    Returns:
        Gross-error sensitivity $\gamma^*_{T|F}$ .

    See Also:
        - [`lmo.contrib.scipy_stats.l_rv_generic.l_moment_influence`
        ][lmo.contrib.scipy_stats.l_rv_generic.l_moment_influence]
        - [`rejection_point`][lmo.diagnostic.rejection_point]

    """
    a, b = domain

    if np.isinf(influence_fn(a)) or np.isinf(influence_fn(b)):
        return np.inf

    def obj(xs: _ArrF8) -> float:
        return -abs(influence_fn(xs[0]))

    bounds = None if np.isneginf(a) and np.isposinf(b) else [(a, b)]

    res = cast(
        lspt.OptimizeResult,
        minimize(
            obj,
            bounds=bounds,
            x0=[min(max(0, a), b)],
            method='COBYLA',
        ),
    )
    if not res.success:
        warnings.warn(
            res.message,
            OptimizeWarning,
            stacklevel=1,
        )

    return -res.fun


def shift_sensitivity(
    influence_fn: Callable[[float], float],
    /,
    domain: _Tuple2[float] = (-math.inf, math.inf),
) -> float:
    r"""
    Evaluate the *local-shift sensitivity* of an influence function
    $\psi_{T|F}(x)$ given a *statistical functional* $T$ (e.g. an L-moment)
    and cumulative distribution function $F(x)$.

    $$
    \lambda^*_{T|F} = \max_{x \neq y}
    \left| \frac{ \psi_{T|F}(y) - \psi_{T|F}(x) }{ y - x } \right|
    $$

    Represents the effect of shifting an observation slightly from $x$, to a
    neighbouring point $y$.
    For instance, adding an observation at $y$ and removing one at $x$.

    Examples:
        Evaluate the local-shift sensitivity of the standard exponential
        distribution's LL-skewness ($\tau^{(0, 1)}_3$) and LL-kurtosis
        ($\tau^{(0, 1)}_4$) coefficients:

        >>> from lmo.diagnostic import shift_sensitivity
        >>> from scipy.stats import expon
        >>> ll_skew_if = expon.l_ratio_influence(3, 2, trim=(0, 1))
        >>> ll_kurt_if = expon.l_ratio_influence(4, 2, trim=(0, 1))
        >>> domain = 0, float('inf')
        >>> shift_sensitivity(ll_skew_if, domain)
        0.837735
        >>> shift_sensitivity(ll_kurt_if, domain)
        1.442062

        Let's compare these with the untrimmed ones:

        >>> shift_sensitivity(expon.l_ratio_influence(3, 2), domain)
        1.920317
        >>> shift_sensitivity(expon.l_ratio_influence(4, 2), domain)
        1.047565

    Args:
        influence_fn: Univariate influence function.
        domain: Domain of the CDF. Defaults to $(-\infty, \infty)$.

    Returns:
        Local-shift sensitivity $\lambda^*_{T|F}$ .

    See Also:
        - [`lmo.contrib.scipy_stats.l_rv_generic.l_moment_influence`
        ][lmo.contrib.scipy_stats.l_rv_generic.l_moment_influence]
        - [`error_sensitivity`][lmo.diagnostic.error_sensitivity]

    References:
        - [Frank R. Hampel (1974) - The Influence Curve and its Role in
            Robust Estimation](https://doi.org/10.2307/2285666)

    """

    def obj(xs: _ArrF8) -> float:
        x, y = xs
        if y == x:
            return 0
        return -abs((influence_fn(y) - influence_fn(x)) / (y - x))

    a, b = domain
    bounds = None if np.isneginf(a) and np.isposinf(b) else [(a, b)]

    res = cast(
        lspt.OptimizeResult,
        minimize(
            obj,
            bounds=bounds,
            x0=[min(max(0, a), b), min(max(1, a), b)],
            method='COBYLA',
        ),
    )
    if not res.success:
        warnings.warn(
            cast(str, res.message),
            OptimizeWarning,
            stacklevel=1,
        )

    return -res.fun
