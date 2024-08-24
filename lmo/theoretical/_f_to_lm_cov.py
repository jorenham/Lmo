from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

import lmo.typing.np as lnpt
import lmo.typing.scipy as lspt
from lmo._poly import eval_sh_jacobi
from lmo._utils import clean_order, clean_trim, moments_to_stats_cov, round0
from ._f_to_lm import l_moment_from_cdf
from ._utils import ALPHA, l_const, nquad, tighten_cdf_support


if TYPE_CHECKING:
    import lmo.typing as lmt


__all__ = ['l_moment_cov_from_cdf', 'l_stats_cov_from_cdf']


_T = TypeVar('_T')

_Pair: TypeAlias = tuple[_T, _T]
_Fn1: TypeAlias = Callable[[float], float | lnpt.Float]
_ArrF8: TypeAlias = npt.NDArray[np.float64]


def l_moment_cov_from_cdf(
    cdf: _Fn1,
    r_max: lmt.AnyOrder,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
) -> _ArrF8:
    r"""
    L-moments that are estimated from $n$ samples of a distribution with CDF
    $F$, converge to the multivariate normal distribution as the sample size
    $n \rightarrow \infty$.

    $$
    \sqrt{n} \left(
        \vec{l}^{(s, t)} - \vec{\lambda}^{(s, t)}
    \right)
    \sim
    \mathcal{N}(
        \vec{0},
        \mathbf{\Lambda}^{(s, t)}
    )
    $$

    Here, $\vec{l}^{(s, t)} = \left[l^{(s, t)}_r, \dots, l^{(s, t)}_{r_{max}}
    \right]^T$ is a vector of estimated sample L-moments,
    and $\vec{\lambda}^{(s, t)}$ its theoretical ("true") counterpart.

    This function calculates the covariance matrix

    $$
    \begin{align*}
    \bf{\Lambda}^{(s,t)}_{k, r}
        &= \mathrm{Cov}[l^{(s, t)}_k, l^{(s, t)}_r] \\
        &= c_k c_r
        \iint\limits_{x < y} \left(
            p^{(s, t)}_k(u) \ p^{(s, t)}_r(v) +
            p^{(s, t)}_r(u) \ p^{(s, t)}_k(v)
        \right) \
        w^{(s + 1,\ t)}(u) \
        w^{(s,\ t + 1)}(v) \
        \mathrm{d} x \
        \mathrm{d} y \;
    ,
    \end{align*}
    $$

    where $u = F_X(x)$ and $v = F_Y(y)$ (marginal) probability integral
    transforms, and

    $$
    c_n = \frac{\Gamma(n) \Gamma(n+s+t+1)}{n \Gamma(n+s) \Gamma(n+t)}\;,
    $$

    the shifted Jacobi polynomial
    $p^{(s, t)}_n(u) = P^{(t, s)}_{n-1}(2u - 1)$, $P^{(t, s)}_m$, and
    $w^{(s, t)}(u) = u^s (1 - u)^t$ its weight function.

    Notes:
        This function uses [`scipy.integrate.nquad`][scipy.integrate.nquad]
        for numerical integration. Unexpected results may be returned if the
        integral does not exist, or does not converge.
        The results are rounded to match the order of magnitude of the
        absolute error of [`scipy.integrate.nquad`][scipy.integrate.nquad].

        This function is not vectorized or parallelized.

        For small sample sizes ($n < 100$), the covariances of the
        higher-order L-moments ($r > 2$) can be biased. But this bias quickly
        disappears at roughly $n > 200$ (depending on the trim- and L-moment
        orders).

    Args:
        cdf:
            Cumulative Distribution Function (CDF), $F_X(x) = P(X \le x)$.
            Must be a continuous monotone increasing function with
            signature `(float) -> float`, whose return value lies in $[0, 1]$.
        r_max:
            The amount of L-moment orders to consider. If for example
            `r_max = 4`, the covariance matrix will be of shape `(4, 4)`, and
            the columns and rows correspond to the L-moments of order
            $r = 1, \dots, r_{max}$.
        trim:
            Left- and right- trim. Must be a tuple of two non-negative ints
            or floats.

    Other parameters:
        support:
            The subinterval of the nonzero domain of `cdf`.
            Generally it's not needed to provide this, as it will be guessed
            automatically.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].

    Returns:
        cov: Covariance matrix, with shape `(r_max, r_max)`.

    Raises:
        RuntimeError: If the covariance matrix is invalid.

    See Also:
        - [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf] -
            Population L-moments from the cumulative distribution function
        - [`l_moment_from_ppf`][lmo.theoretical.l_moment_from_ppf] -
            Population L-moments from the quantile function
        - [`lmo.l_moment`][lmo.l_moment] - Unbiased L-moment estimation from
            samples
        - [`lmo.l_moment_cov`][lmo.l_moment_cov] - Distribution-free exact
            L-moment exact covariance estimate.

    References:
        - [J.R.M. Hosking (1990) - L-moments: Analysis and Estimation of
            Distributions Using Linear Combinations of Order Statistics
            ](https://jstor.org/stable/2345653)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
    """
    rs = clean_order(r_max, 'rmax', 0)
    if rs == 0:
        return np.empty((0, 0))

    s, t = clean_trim(trim)

    _cdf = functools.cache(cdf)

    if support is None:
        a, b = tighten_cdf_support(_cdf, (-np.inf, np.inf))
    else:
        a, b = map(float, support)

    c_n = np.array([l_const(n + 1, s, t) for n in range(rs)])

    def integrand(x: float, y: float, k: int, r: int) -> float:
        u, v = _cdf(x), _cdf(y)
        return (
            c_n[k]
            * c_n[r]
            * (
                (
                    eval_sh_jacobi(k, t, s, u) * eval_sh_jacobi(r, t, s, v)
                    + eval_sh_jacobi(r, t, s, u) * eval_sh_jacobi(k, t, s, v)
                )
                * u
                * (1 - v)
                * (u * v) ** s
                * ((1 - u) * (1 - v)) ** t
            )
        )

    def range_x(y: float, *_: int) -> tuple[float, float]:
        return (a, y)

    cov = np.empty((rs, rs), dtype=np.float64)
    for k, r in zip(*np.triu_indices(rs), strict=True):
        cov_kr = nquad(
            integrand,
            [(a, b), range_x],
            quad_opts,
            k,
            r,
        )
        if k == r and cov_kr <= 0:
            msg = f'negative variance encountered at {r}: {cov_kr}'
            raise RuntimeError(msg)

        cov[k, r] = cov[r, k] = cov_kr

    # Validate the Cauchy-Schwartz inequality
    cov_max = np.sqrt(np.outer(cov.diagonal(), cov.diagonal()))
    invalid = np.abs(cov) > cov_max
    if np.any(invalid):
        invalid_kr = list(np.argwhere(invalid)[0])
        msg = (
            f'invalid covariance matrix: Cauchy-Schwartz inequality violated '
            f'at {invalid_kr}: \n{cov}'
        )
        raise RuntimeError(msg)

    return round0(cov)


def l_stats_cov_from_cdf(
    cdf: _Fn1,
    /,
    num: lmt.AnyOrder = 4,
    trim: lmt.AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: _Fn1 | None = None,
) -> _ArrF8:
    r"""
    Similar to [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf], but
    for the [`lmo.l_stats`][lmo.l_stats].

    As the sample size $n \rightarrow \infty$, the L-moment ratio's are also
    distributed (multivariate) normally. The L-stats are defined to be
    L-moments for $r\le 2$, and L-ratio coefficients otherwise.

    The corresponding covariance matrix has been found to be

    $$
    \bf{T}^{(s, t)}_{k, r} =
    \begin{cases}
        \bf{\Lambda}^{(s, t)}_{k, r}
            & k \le 2 \wedge r \le 2 \\
        \frac{
            \bf{\Lambda}^{(s, t)}_{k, r}
            - \tau_r \bf{\Lambda}^{(s, t)}_{k, 2}
        }{
            \lambda^{(s,t)}_{2}
        }
            & k \le 2 \wedge r > 2 \\
        \frac{
            \bf{\Lambda}^{(s, t)}_{k, r}
            - \tau_k \bf{\Lambda}^{(s, t)}_{2, r}
            - \tau_r \bf{\Lambda}^{(s, t)}_{k, 2}
            + \tau_k \tau_r \bf{\Lambda}^{(s, t)}_{2, 2}
        }{
            \Big( \lambda^{(s,t)}_{2} \Big)^2
        }
            & k > 2 \wedge r > 2
    \end{cases}
    $$

    where $\bf{\Lambda}^{(s, t)}$ is the covariance matrix of the L-moments
    from [`l_moment_cov_from_cdf`][lmo.theoretical.l_moment_cov_from_cdf],
    and $\tau^{(s,t)}_r = \lambda^{(s,t)}_r / \lambda^{(s,t)}_2$ the
    population L-ratio.

    Args:
        cdf:
            Cumulative Distribution Function (CDF), $F_X(x) = P(X \le x)$.
            Must be a continuous monotone increasing function with
            signature `(float) -> float`, whose return value lies in $[0, 1]$.
        num:
            The amount of L-statistics to return. Defaults to 4.
        trim:
            Left- and right- trim. Must be a tuple of two non-negative ints
            or floats.

    Other parameters:
        support:
            The subinterval of the nonzero domain of `cdf`.
            Generally it's not needed to provide this, as it will be guessed
            automatically.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        alpha: Two-sided quantile to split the integral at.
        ppf: Quantile function, for calculating the split integral limits.

    References:
        - [J.R.M. Hosking (1990) - L-moments: Analysis and Estimation of
            Distributions Using Linear Combinations of Order Statistics
            ](https://jstor.org/stable/2345653)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
    """
    rs = clean_order(num, 'num', 0)
    ll_kr = l_moment_cov_from_cdf(
        cdf,
        rs,
        trim,
        support=support,
        quad_opts=quad_opts,
    )
    if rs <= 2:
        return ll_kr

    l_2r = l_moment_from_cdf(
        cdf,
        np.arange(2, rs + 1),
        trim=trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
        ppf=ppf,
    )
    t_0r = np.r_[1, 0, l_2r] / l_2r[0]

    return round0(moments_to_stats_cov(t_0r, ll_kr))
