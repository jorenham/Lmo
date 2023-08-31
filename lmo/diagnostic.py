"""Statistical test and tools."""

__all__ = (
    'normaltest',
    'l_moment_gof',
    'l_moment_bounds',
    'l_ratio_bounds',
)

from collections.abc import Callable
from math import lgamma
from typing import Any, NamedTuple, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
from scipy.special import chdtrc  # type: ignore

from . import theoretical
from ._lm import l_ratio
from ._utils import clean_orders, clean_trim
from .typing import AnyFloat, AnyInt, AnyTrim, IntVector

T = TypeVar('T', bound=np.floating[Any])


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

    statistic: float | npt.NDArray[np.float_]
    pvalue: float | npt.NDArray[np.float_]

    @property
    def is_valid(self) -> bool | npt.NDArray[np.bool_]:
        """Check if the statistic is finite and not `nan`."""
        return np.isfinite(self.statistic)

    def is_significant(
        self,
        level: float = 0.05,
        /,
    ) -> bool | npt.NDArray[np.bool_]:
        """
        Whether or not the null hypothesis can be rejected, with a certain
        confidence level (5% by default).
        """
        if not (0 < level < 1):
            msg = 'significance level must lie between 0 and 1'
            raise ValueError(msg)
        return self.pvalue < level


def normaltest(
    a: npt.ArrayLike,
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
        0.04806618...
        >>> normaltest_scipy(x)[1]
        0.08435627...

        At a 5% significance level, Lmo's test is signficiant (i.e. indicating
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
    t3, t4 = l_ratio(a, [3, 4], [2, 2], axis=axis)

    # theoretical L-skew and L-kurtosis of the normal distribution (for all
    # loc/mu and scale/sigma)
    tau3, tau4 = 0.0, 30 / np.pi * np.arctan(np.sqrt(2)) - 9

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


def _gof_chi2(
    err: npt.NDArray[np.floating[Any]],
    cov: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    prec = np.linalg.inv(cov)  # precision matrix
    stat = np.asarray(err.T @ prec @ err)
    pval = cast(npt.NDArray[np.float_], chdtrc(err.shape[0], stat))
    return stat, pval


def l_moment_gof(
    l_moments: npt.NDArray[np.float_],
    n_obs: int,
    cdf: Callable[[float], float],
    /,
    trim: AnyTrim = (0, 0),
    support: tuple[AnyFloat, AnyFloat] = (-np.inf, np.inf),
    **kwargs: Any,
) -> HypothesisTestResult:
    r"""
    Goodness-of-fit (GOF) hypothesis test for the null hypothesis that the
    observed L-moments come from a distribution with the given cumulative
    distribution function (CDF).

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

    See Also:
        - [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf]
        - ['l_moment_cov_from_cdf'][lmo.theoretical.l_moment_cov_from_cdf]

    Todo:
        - Vectorize the `l_moments` parameter (axis=0 only).
        - Add a `nan_policy: 'omit' | 'raise' | 'propagate'` parameter,
            and apply to both the sample- and theoretical L-moments.

    """
    l_r = np.asarray_chkfinite(l_moments)

    if l_r.ndim > 1:
        msg = f'l_moments must be 1D, shape is {l_r.shape}'
        raise TypeError(msg)
    if (n := len(l_r)) < 2:
        msg = f'at least 2 L-moments are required, got {n}'
        raise TypeError(msg)
    if n_obs <= n:
        msg = f'n_obs must be >{n}, got {n_obs}'
        raise ValueError(msg)

    r = np.arange(1, 1 + n)

    lambda_r = theoretical.l_moment_from_cdf(
        cdf,
        r,
        trim=trim,
        support=support,
        **kwargs,
    )

    nanstat = np.nan * lambda_r[0]
    nanresult = HypothesisTestResult(nanstat, nanstat)

    if lambda_r[1] <= 0 or not np.all(np.isfinite(lambda_r)):
        # avoid evaluating the (likely to be incorrect) covariance matrix:
        # return nan(s) (with correct shape)
        return nanresult

    if n > 2:
        # ensure the L-ratio's are within the outermost bounds.
        tau_r = lambda_r[2:] / lambda_r[1]
        bounds = l_ratio_bounds(r[2:], trim=trim, has_variance=False)
        if np.any(abs(tau_r) > bounds):
            return nanresult

    lambda_rr = theoretical.l_moment_cov_from_cdf(
        cdf,
        n,
        trim=trim,
        support=support,
        **kwargs,
    )
    if not (
        np.all(lambda_rr.diagonal() > 0)
        and np.all(np.isfinite(lambda_rr))
        and np.all(np.linalg.eigvalsh(lambda_rr) > 0)  # positive definite
    ):
        return nanresult

    return HypothesisTestResult(*_gof_chi2(l_r - lambda_r, lambda_rr / n_obs))



def _lm2_bounds_single(r: int, trim: tuple[float, float]) -> float:
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
    Callable[[IntVector, tuple[float, float]], npt.NDArray[np.float_]],
    np.vectorize(
        _lm2_bounds_single,
        otypes=[float],
        excluded={1},
        signature='()->()',
    ),
)


@overload
def l_moment_bounds(
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    scale: float = ...,
) -> npt.NDArray[np.float_]:
    ...


@overload
def l_moment_bounds(
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    scale: float = ...,
) -> float:
    ...


def l_moment_bounds(
    r: IntVector | AnyInt,
    /,
    trim: AnyTrim = (0, 0),
    scale: float = 1.0,
) -> float | npt.NDArray[np.float_]:
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
    _r = clean_orders(r, rmin=1)
    _trim = clean_trim(trim)
    return scale * np.sqrt(_lm2_bounds(_r, _trim))[()]


@overload
def l_ratio_bounds(
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    has_variance: bool = ...,
) -> npt.NDArray[np.float_]:
    ...


@overload
def l_ratio_bounds(
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    has_variance: bool = ...,
) -> float:
    ...


def l_ratio_bounds(
    r: IntVector | AnyInt,
    /,
    trim: AnyTrim = (0, 0),
    *,
    has_variance: bool = True,
) -> float | npt.NDArray[np.float_]:
    r"""
    Returns the absolute upper bounds $T^{(s,t)}_r$ on L-moment ratio's
    $\tau^{(s,t)}_r = \lambda^{(s,t)}_r / \lambda^{(s,t)}_r$, for $r \ge 2$.
    So $\left| \tau^{(s,t)}_r(X) \right| \le T^{(s,t)}_r$, given that
    $\mathrm{Var}[X] = \sigma^2_X$ exists.

    If the variance of the distribution is not defined, e.g. in case of the
    [Cauchy distribution](https://wikipedia.org/wiki/Cauchy_distribution),
    this method will not work. In this case, the looser bounds from
    Hosking (2007) can be used instead, by passing `has_variance=False`.

    Examples:
        Calculate the bounds for different degrees of trimming:

        >>> l_ratio_bounds([1, 2, 3, 4])
        array([       inf, 1.        , 0.77459667, 0.65465367])
        >>> # the bounds for r=1,2 are the same for all trimmings; skip them
        >>> l_ratio_bounds([3, 4], trim=(1, 1))
        array([0.61475926, 0.4546206 ])
        >>> l_ratio_bounds([3, 4], trim=(.5, 1.5))
        array([0.65060005, 0.49736473])

        In case of undefined variance, the bounds become a lot looser:

        >>> l_ratio_bounds([3, 4], has_variance=False)
        array([1., 1.])
        >>> l_ratio_bounds([3, 4], trim=(1, 1), has_variance=False)
        array([1.11111111, 1.25      ])
        >>> l_ratio_bounds([3, 4], trim=(.5, 1.5), has_variance=False)
        array([1.33333333, 1.71428571])

    Args:
        r: Order of the L-moment ratio(s), as positive integer scalar or
            array-like.
        trim: Tuple of left- and right- trim-lengths, matching those of the
            relevant L-moment ratio's.
        has_variance:
            Set to False if the distribution has undefined variance, in which
            case the (looser) bounds from J.R.M. Hosking (2007) will be used.

    Returns:
        Array or scalar with shape like $r$.

    See Also:
        - [`l_ratio`][lmo.l_ratio]
        - [`l_ratio_se`][lmo.l_ratio_se]

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
        L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    """
    if has_variance:
        return l_moment_bounds(r, trim) / l_moment_bounds(2, trim)

    # otherwise, fall back to the (very) loose bounds from Hosking
    _r = clean_orders(r, rmin=1)
    _n = np.max(_r) + 1

    s, t = clean_trim(trim)

    out = np.ones(_n)
    if _n > 1:
        # `L-loc / L-scale (= 1 / CV)` is unbounded
        out[1] = np.inf

    if _n > 3 and s and t:
        # if not trimmed, then the bounds are simply 1
        p, q = s + t, min(s, t)
        for _k in range(3, _n):
            out[_k] = out[_k - 1] * (1 + p / _k) / (1 + q / (_k - 1))

    return out[_r]
