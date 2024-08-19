"""
Theoretical (population) L-moments of known univariate probability
distributions.
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from math import exp, factorial, gamma, lgamma, log
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Final,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
import scipy.integrate as spi

from ._poly import eval_sh_jacobi
from ._utils import (
    clean_order,
    clean_orders,
    clean_trim,
    l_stats_orders,
    moments_to_ratio,
    moments_to_stats_cov,
    plotting_positions,
    round0,
)
from .special import fourier_jacobi, fpow


if TYPE_CHECKING:
    import optype.numpy as onpt

    import lmo.typing.np as lnpt
    import lmo.typing.scipy as lspt
    from .typing import AnyOrder, AnyOrderND, AnyTrim


__all__ = (
    'l_moment_from_cdf',
    'l_moment_from_ppf',
    'l_moment_from_qdf',
    'l_ratio_from_cdf',
    'l_ratio_from_ppf',
    'l_stats_from_cdf',
    'l_stats_from_ppf',

    'l_comoment_from_pdf',
    'l_coratio_from_pdf',

    'l_moment_cov_from_cdf',
    'l_stats_cov_from_cdf',

    'l_moment_influence_from_cdf',
    'l_ratio_influence_from_cdf',

    'entropy_from_qdf',

    'ppf_from_l_moments',
    'qdf_from_l_moments',
    'cdf_from_ppf',
)

_T = TypeVar('_T')
_T_x = TypeVar('_T_x', bound=float | npt.NDArray[np.float64])
_Tss = ParamSpec('_Tss')

_Pair: TypeAlias = tuple[_T, _T]
_Fn1: TypeAlias = Callable[[float], float]
_ArrF8: TypeAlias = npt.NDArray[np.float64]

ALPHA: Final[float] = 0.1
QUAD_LIMIT: Final[int] = 100

# pyright: reportUnknownMemberType=false


def _nquad(
    integrand: Callable[Concatenate[float, float, _Tss], float],
    domains: Sequence[_Pair[float] | Callable[..., _Pair[float]]],
    opts: lspt.QuadOptions | None = None,
    *args: _Tss.args,
    **kwds: _Tss.kwargs,
) -> float:
    # nquad only has an `args` param for some invalid reason
    fn = functools.partial(integrand, **kwds) if kwds else integrand

    return cast(
        tuple[float, float],
        spi.nquad(fn, domains[::-1], args, opts=opts),
    )[0]


@functools.cache
def _l_moment_const(r: int, s: float, t: float, k: int = 0) -> float:
    assert k >= 0

    if r <= k:
        return 1.0
    if s == t == 0:
        if k == 0:
            return 1.0
        if k == 1:
            return 1 / (r - 1)

    # math.lgamma is faster (and has better type annotations) than
    # scipy.special.loggamma.
    if r + s + t <= 20:
        v = gamma(r + s + t + 1) / (gamma(r + s) * gamma(r + t))
    elif r + s + t <= 128:
        v = exp(lgamma(r + s + t + 1) - lgamma(r + s) - lgamma(r + t))
    else:
        return exp(
            + lgamma(r + s + t + 1)
            - lgamma(r + s)
            - lgamma(r + t)
            + lgamma(r - k)
            - log(r),
        )

    return factorial(r - 1 - k) / r * v


def _tighten_cdf_support(
    cdf: _Fn1,
    support: _Pair[float] | None = None,
) -> _Pair[float]:
    """Attempt to tighten the support by checking some common bounds."""
    a, b = (-np.inf, np.inf) if support is None else map(float, support)

    # assert a < b, (a, b)
    # assert (u_a := cdf(a)) == 0, (a, u_a)
    # assert (u_b := cdf(b)) == 1, (b, u_b)

    # attempt to tighten the default support by checking some common bounds
    if cdf(0) == 0:
        # left-bounded at 0 (e.g. weibull)
        a = 0

        if (u1 := cdf(1)) == 0:
            # left-bounded at 1 (e.g. pareto)
            a = 1
        elif u1 == 1:
            # right-bounded at 1 (e.g. beta)
            b = 1

    return a, b


@overload
def l_moment_from_cdf(
    cdf: _Fn1,
    r: AnyOrderND,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] | None = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
    ppf: _Fn1 | None = ...,
) -> _ArrF8: ...

@overload
def l_moment_from_cdf(
    cdf: _Fn1,
    r: AnyOrder,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] | None = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
    ppf: _Fn1 | None = ...,
) -> np.float64: ...


def l_moment_from_cdf(
    cdf: _Fn1,
    r: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: _Fn1 | None = None,
) -> np.float64 | _ArrF8:
    r"""
    Evaluate the population L-moment of a continuous probability distribution,
    using its Cumulative Distribution Function (CDF) $F_X(x) = P(X \le x)$.

    $$
    \lambda^{(s, t)}_r =
    \begin{cases}
        1 & r = 0 \\
        \displaystyle
        \int_{\mathbb{R}}
            \left(H(x) - I_u(s + 1,\ t + 1)\right) \
            \mathrm{d} x
        & r = 1 \\
        \displaystyle
        \frac{c^{(s,t)}_r}{r}
        \int_{\mathbb{R}}
            u^{s + 1}
            \left(1 - u\right)^{t + 1} \
            \widetilde{P}^{(t + 1, s + 1)}_{r - 2}(u) \
            \mathrm{d} x
        & r > 1 \;
    ,
    \end{cases}
    $$

    where,

    $$
    c^{(s,t)}_r =
        \frac{r + s + t}{r}
        \frac{\B(r,\ r + s + t)}{\B(r + s,\ r + t)} \;
    ,
    $$

    $\widetilde{P}^{(\alpha, \beta)}_k(x)$ the shifted ($x \mapsto 2x-1$)
    Jacobi polynomial, $H(x)$ the Heaviside step function, and
    $I_x(\alpha, \beta)$ the regularized incomplete gamma function, and
    $u = F_X(x)$ the probability integral transform of $x \sim X$.

    Notes:
        Numerical integration is performed with
        [`scipy.integrate.quad`][scipy.integrate.quad], which cannot verify
        whether the integral exists and is finite. If it returns an error
        message, an `IntegrationWarning` is issues, and `nan` is returned
        (even if `quad` returned a finite result).

    Examples:
        Evaluate the first 4 L- and TL-moments of the standard normal
        distribution:

        >>> from scipy.special import ndtr  # standard normal CDF
        >>> l_moment_from_cdf(ndtr, [1, 2, 3, 4])
        array([0.        , 0.56418958, 0.        , 0.06917061])
        >>> l_moment_from_cdf(ndtr, [1, 2, 3, 4], trim=1)
        array([0.        , 0.29701138, 0.        , 0.01855727])

        Evaluate the first 4 TL-moments of the standard Cauchy distribution:

        >>> def cdf_cauchy(x: float) -> float:
        ...     return np.arctan(x) / np.pi + 1 / 2
        >>> l_moment_from_cdf(cdf_cauchy, [1, 2, 3, 4], trim=1)
        array([0.        , 0.69782723, 0.        , 0.23922105])

    Args:
        cdf:
            Cumulative Distribution Function (CDF), $F_X(x) = P(X \le x)$.
            Must be a continuous monotone increasing function with
            signature `(float) -> float`, whose return value lies in $[0, 1]$.
        r:
            L-moment order(s), non-negative integer or array-like of integers.
        trim:
            Left- and right- trim, either as a $(s, t)$ tuple with
            $s, t > -1/2$, or $t$ as alias for $(t, t)$.

    Other parameters:
        support: The subinterval of the nonzero domain of `cdf`.
            Generally it's not needed to provide this, as it will be guessed
            automatically.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        alpha:
            Split the integral into integrals with limits
            $[a, F^{-1}(\alpha)]$, $[F(\alpha), F^{-1}(1 - \alpha)]$ and
            $[F^{-1}(1 - \alpha), b]$ to improve numerical stability.
            So $\alpha$ can be consideresd the size of
            the tail. Numerical experiments have found 0.05 to give good
            results for different distributions.
        ppf:
            The inverse of the cdf, used with `alpha` to calculate the
            integral split points (if provided).

    Raises:
        TypeError: `r` is not integer-valued or negative
        ValueError: `r` is negative

    Returns:
        lmbda:
            The population L-moment(s), a scalar or float array like `r`.
            If `nan`, consult the related `IntegrationWarning` message.

    References:
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    See Also:
        - [`theoretical.l_moment_from_ppf`][lmo.theoretical.l_moment_from_ppf]:
          population L-moment, using the inverse CDF
        - [`l_moment`][lmo.l_moment]: sample L-moment

    """
    rs = clean_orders(np.asanyarray(r))
    s, t = clean_trim(trim)

    from scipy.special import betainc

    def integrand(x: float, _r: int) -> float:
        p = cdf(x)
        if _r == 1:
            if s or t:  # noqa: SIM108
                v = cast(float, betainc(s + 1, t + 1, p))
            else:
                v = p
            return np.heaviside(x, .5) - v

        return (
            np.sqrt(2 * _r - 1)
            * p ** (s + 1) * (1 - p) ** (t + 1)
            * eval_sh_jacobi(_r - 2, t + 1, s + 1, p)
        )

    a, d = support or _tighten_cdf_support(cdf, support)
    b, c = (ppf(alpha), ppf(1 - alpha)) if ppf else (a, d)

    loc0 = a if np.isfinite(a) and a > 0 else 0

    kwds = quad_opts or {}
    _ = kwds.setdefault('limit', QUAD_LIMIT)

    def _l_moment_single(_r: int) -> float:
        if _r == 0:
            return 1

        return _l_moment_const(_r, s, t, 1) * cast(
            float,
            (spi.quad(integrand, a, b, (_r,), **kwds)[0] if a < b else 0) +
            spi.quad(integrand, b, c, (_r,), **kwds)[0] +
            (spi.quad(integrand, c, d, (_r,), **kwds)[0] if c < d else 0),
        ) / np.sqrt(2 * _r - 1) + loc0 * (_r == 1)

    l_r_cache: dict[int, float] = {}
    l_r = np.empty_like(rs, dtype=np.float64)
    for i, _r in np.ndenumerate(rs):
        _k = int(_r)
        if _k in l_r_cache:
            l_r[i] = l_r_cache[_k]
        else:
            l_r[i] = l_r_cache[_k] = _l_moment_single(_k)

    return round0(l_r, 1e-12)[()]  # convert back to scalar if needed


@overload
def l_moment_from_ppf(
    ppf: _Fn1,
    r: AnyOrderND,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
) -> _ArrF8: ...

@overload
def l_moment_from_ppf(
    ppf: _Fn1,
    r: AnyOrder,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float64: ...


def l_moment_from_ppf(
    ppf: _Fn1,
    r: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
) -> np.float64 | _ArrF8:
    r"""
    Evaluate the population L-moment of a univariate probability distribution,
    using its Percentile Function (PPF), $x(F)$, also commonly known as the
    quantile function, which is the inverse of the Cumulative Distribution
    Function (CDF).

    $$
    \lambda^{(s, t)}_r =
        c^{(s, t)}_r
        \int_0^1
            F^s (1 - F)^t \
            \widetilde{P}^{(t, s)}_{r - 1}(F) \
            x(F) \
            \mathrm{d} F \;
    ,
    $$

    where

    $$
    c^{(s,t)}_r =
    \frac{r+s+t}{r}
    \frac{B(r,\,r+s+t)}{B(r+s,\,r+t)} \;,
    $$

    and $\widetilde{P}^{(\alpha, \beta)}_k(x)$ the shifted ($x \mapsto 2x-1$)
    Jacobi polynomial.

    Notes:
        Numerical integration is performed with
        [`scipy.integrate.quad`][scipy.integrate.quad], which cannot verify
        whether the integral exists and is finite. If it returns an error
        message, an `IntegrationWarning` is issues, and `nan` is returned
        (even if `quad` returned a finite result).

    Examples:
        Evaluate the L- and TL-location and -scale of the standard normal
        distribution:

        >>> from scipy.special import ndtri  # standard normal inverse CDF
        >>> l_moment_from_ppf(ndtri, [1, 2])
        array([0.        , 0.56418958])
        >>> l_moment_from_ppf(ndtri, [1, 2], trim=1)
        array([0.        , 0.29701138])

    Args:
        ppf:
            The quantile function $x(F)$, a monotonically continuous
            increasing function with signature `(float) -> float`, that maps a
            probability in $[0, 1]$, to the domain of the distribution.
        r:
            L-moment order(s), non-negative integer or array-like of integers.
            E.g. 0 gives 1, 1 the L-location, 2 the L-scale, etc.
        trim:
            Left- and right- trim, either as a $(s, t)$ tuple with
            $s, t > -1/2$, or $t$ as alias for $(t, t)$.

    Other parameters:
        support:
            Integration limits. Defaults to (0, 1), as it should. There is no
            need to change this to anything else, and only exists to make the
            function signature consistent with the `*_from_cdf` analogue.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        alpha:
            Split the integral into integrals with limits $[0, \alpha]$,
            $[\alpha, 1-\alpha]$ and $[1-\alpha, 0]$ to improve numerical
            stability. So $\alpha$ can be consideresd the size of the tail.
            Numerical experiments have found 0.1 to give good results for
            different distributions.

    Raises:
        TypeError: Invalid `r` or `trim` types.
        ValueError: Invalid `r` or `trim` values.

    Returns:
        lmbda:
            The population L-moment(s), a scalar or float array like `r`.
            If `nan`, consult the related `IntegrationWarning` message.

    References:
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    See Also:
        - [`theoretical.l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf]:
          population L-moment, using the CDF (i.e. the inverse PPF)
        - [`l_moment`][lmo.l_moment]: sample L-moment
    """
    rs = clean_orders(np.asanyarray(r))
    s, t = clean_trim(trim)

    def integrand(p: float, _r: int) -> float:
        return p**s * (1 - p) ** t * eval_sh_jacobi(_r - 1, t, s, p) * ppf(p)

    quad_kwds = quad_opts or {}
    _ = quad_kwds.setdefault('limit', QUAD_LIMIT)

    def _l_moment_single(_r: int) -> float:
        if _r == 0:
            return 1

        a, b, c, d = support[0], alpha, 1 - alpha, support[1]
        return _l_moment_const(_r, s, t) * cast(
            float,
            spi.quad(integrand, a, b, (_r,), **quad_kwds)[0] +
            spi.quad(integrand, b, c, (_r,), **quad_kwds)[0] +
            spi.quad(integrand, c, d, (_r,), **quad_kwds)[0],
        )

    l_r_cache: dict[int, float] = {}
    l_r = np.empty_like(rs, dtype=np.float64)
    for i, _r in np.ndenumerate(rs):
        _k = int(_r)
        if _k in l_r_cache:
            l_r[i] = l_r_cache[_k]
        else:
            l_r[i] = l_r_cache[_k] = _l_moment_single(_k)

    return round0(l_r)[()]  # convert back to scalar if needed


@overload
def l_moment_from_qdf(
    qdf: _Fn1,
    r: AnyOrderND,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
) -> _ArrF8: ...

@overload
def l_moment_from_qdf(
    qdf: _Fn1,
    r: AnyOrder,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float64: ...


def l_moment_from_qdf(
    qdf: _Fn1,
    r: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
) -> np.float64 | _ArrF8:
    r"""
    Evaluate the population L-moments \( \tlmoment{s, t}{r} \) for \( r > 1 \)
    from the quantile distribution function (QDF), which is the derivative of
    the PPF (quantile function).
    """
    r_qd = clean_orders(np.asanyarray(r), rmin=2)[()]
    r_pp = r_qd - 1

    s_qd, t_qd = clean_trim(trim)
    s_pp, t_pp = s_qd + 1, t_qd + 1

    out = l_moment_from_ppf(
        qdf,
        r_pp,
        trim=(s_pp, t_pp),
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
    )

    # correction of the L-moment constant, which follows from the recurrence
    # relations of the gamma (or beta) functions
    c_scale = r_pp / (r_qd * (r_pp + s_pp + t_pp))

    return out * c_scale


@overload
def l_ratio_from_cdf(
    cdf: _Fn1,
    r: AnyOrderND,
    s: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] | None = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
    ppf: _Fn1 | None = ...,
) -> _ArrF8: ...

@overload
def l_ratio_from_cdf(
    cdf: _Fn1,
    r: AnyOrder | AnyOrderND,
    s: AnyOrderND,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] | None = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
    ppf: _Fn1 | None = ...,
) -> _ArrF8: ...

@overload
def l_ratio_from_cdf(
    cdf: _Fn1,
    r: AnyOrder,
    s: AnyOrder,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] | None = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float64: ...


def l_ratio_from_cdf(
    cdf: _Fn1,
    r: AnyOrder | AnyOrderND,
    s: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: _Fn1 | None = None,
) -> np.float64 | _ArrF8:
    """
    Population L-ratio's from a CDF.

    See Also:
        - [`l_ratio_from_ppf`][lmo.theoretical.l_ratio_from_ppf]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))
    l_rs = l_moment_from_cdf(
        cdf,
        rs,
        trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
        ppf=ppf,
    )
    return moments_to_ratio(rs, l_rs)


@overload
def l_ratio_from_ppf(
    ppf: _Fn1,
    r: AnyOrderND,
    s: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
) -> _ArrF8: ...

@overload
def l_ratio_from_ppf(
    ppf: _Fn1,
    r: AnyOrder | AnyOrderND,
    s: AnyOrderND,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
) -> _ArrF8: ...

@overload
def l_ratio_from_ppf(
    ppf: _Fn1,
    r: AnyOrder,
    s: AnyOrder,
    /,
    trim: AnyTrim = ...,
    *,
    support: _Pair[float] = ...,
    quad_opts: lspt.QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float64:
    ...


def l_ratio_from_ppf(
    ppf: _Fn1,
    r: AnyOrder | AnyOrderND,
    s: AnyOrder | AnyOrderND,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
) -> np.float64 | _ArrF8:
    """
    Population L-ratio's from a PPF.

    See Also:
        - [`l_ratio_from_cdf`][lmo.theoretical.l_ratio_from_cdf]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))
    l_rs = l_moment_from_ppf(
        ppf,
        rs,
        trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
    )
    return moments_to_ratio(rs, l_rs)


def l_stats_from_cdf(
    cdf: _Fn1,
    num: int = 4,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: _Fn1 | None = None,
) -> _ArrF8:
    r"""
    Calculates the theoretical- / population- L-moments (for $r \le 2$)
    and L-ratio's (for $r > 2$) of a distribution, from its CDF.

    By default, the first `num = 4` population L-stats are calculated:

    - $\lambda^{(s,t)}_1$ - *L-loc*ation
    - $\lambda^{(s,t)}_2$ - *L-scale*
    - $\tau^{(s,t)}_3$ - *L-skew*ness coefficient
    - $\tau^{(s,t)}_4$ - *L-kurt*osis coefficient

    This function is equivalent to
    `l_ratio_from_cdf(cdf, [1, 2, 3, 4], [0, 0, 2, 2], *, **)`.

    Note:
        This should not be confused with the term *L-statistic*, which is
        sometimes used to describe any linear combination of order statistics.

    See Also:
        - [`l_stats_from_ppf`][lmo.theoretical.l_stats_from_ppf] - Population
            L-stats from the quantile function.
        - [`l_ratio_from_cdf`][lmo.theoretical.l_ratio_from_cdf] - Generalized
            population L-ratio's from the CDF.
        - [`lmo.l_stats`][lmo.l_stats] - Unbiased sample estimation of L-stats.

    """
    return l_ratio_from_cdf(
        cdf,
        *l_stats_orders(num),
        trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
        ppf=ppf,
    )


def l_stats_from_ppf(
    ppf: _Fn1,
    num: int = 4,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
) -> _ArrF8:
    r"""
    Calculates the theoretical- / population- L-moments (for $r \le 2$)
    and L-ratio's (for $r > 2$) of a distribution, from its quantile function.

    By default, the first `num = 4` population L-stats are calculated:

    - $\lambda^{(s,t)}_1$ - *L-loc*ation
    - $\lambda^{(s,t)}_2$ - *L-scale*
    - $\tau^{(s,t)}_3$ - *L-skew*ness coefficient
    - $\tau^{(s,t)}_4$ - *L-kurt*osis coefficient

    This function is equivalent to
    `l_ratio_from_cdf(cdf, [1, 2, 3, 4], [0, 0, 2, 2], *, **)`.

    Note:
        This should not be confused with the term *L-statistic*, which is
        sometimes used to describe any linear combination of order statistics.

    See Also:
        - [`l_stats_from_cdf`][lmo.theoretical.l_stats_from_cdf] - Population
            L-stats from the CDF.
        - [`l_ratio_from_ppf`][lmo.theoretical.l_ratio_from_ppf] - Generalized
            population L-ratio's from the quantile function.
        - [`lmo.l_stats`][lmo.l_stats] - Unbiased sample estimation of L-stats.
    """
    return l_ratio_from_ppf(
        ppf,
        *l_stats_orders(num),
        trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
    )


def l_moment_cov_from_cdf(
    cdf: _Fn1,
    r_max: AnyOrder,
    /,
    trim: AnyTrim = 0,
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
        a, b = _tighten_cdf_support(_cdf, (-np.inf, np.inf))
    else:
        a, b = map(float, support)

    c_n = np.array([_l_moment_const(n + 1, s, t) for n in range(rs)])

    def integrand(x: float, y: float, k: int, r: int) -> float:
        u, v = _cdf(x), _cdf(y)
        return c_n[k] * c_n[r] * (
            (
                eval_sh_jacobi(k, t, s, u)
                * eval_sh_jacobi(r, t, s, v)
                + eval_sh_jacobi(r, t, s, u)
                * eval_sh_jacobi(k, t, s, v)
            )
            * u * (1 - v)
            * (u * v)**s * ((1 - u) * (1 - v))**t
        )

    def range_x(y: float, *_: int) -> tuple[float, float]:
        return (a, y)

    cov = np.empty((rs, rs), dtype=np.float64)
    for k, r in zip(*np.triu_indices(rs), strict=True):
        cov_kr = _nquad(
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
    num: AnyOrder = 4,
    trim: AnyTrim = 0,
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


def l_moment_influence_from_cdf(
    cdf: Callable[[_ArrF8], _ArrF8],
    r: AnyOrder,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    l_moment: float | np.float64 | None = None,
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
    tol: float = 1e-8,
) -> Callable[[_T_x], _T_x]:
    r"""
    Influence Function (IF) of a theoretical L-moment.

    $$
    \psi_{\lambda^{(s, t)}_r | F}(x)
        = c^{(s,t)}_r
        \, F(x)^s
        \, \big( 1-{F}(x) \big)^t
        \, \tilde{P}^{(s,t)}_{r-1} \big( F(x) \big)
        \, x
        - \lambda^{(s,t)}_r
        \;,
    $$

    with $F$ the CDF, $\tilde{P}^{(s,t)}_{r-1}$ the shifted Jacobi polynomial,
    and

    $$
    c^{(s,t)}_r
        = \frac{r+s+t}{r} \frac{B(r, \, r+s+t)}{B(r+s, \, r+t)}
        \;,
    $$

    where $B$ is the (complete) Beta function.

    The proof is trivial, because population L-moments are
    [linear functionals](https://wikipedia.org/wiki/Linear_form).

    Notes:
        The order parameter `r` is not vectorized.

    Args:
        cdf: Vectorized cumulative distribution function (CDF).
        r: The L-moment order. Must be a non-negative integer.
        trim: Left- and right- trim lengths. Defaults to (0, 0).

    Other parameters:
        support:
            The subinterval of the nonzero domain of `cdf`.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        l_moment:
            The relevant L-moment to use. If not provided, it is calculated
            from the CDF.
        alpha: Two-sided quantile to split the integral at.
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The influence function, with vectorized signature `() -> ()`.

    See Also:
        - [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf]
        - [`lmo.l_moment`][lmo.l_moment]

    """
    _r = clean_order(int(r))
    if _r == 0:
        def influence0(x: _T_x, /) -> _T_x:
            """
            L-moment Influence Function for `r=0`.

            Args:
                x: Scalar or array-like of sample observarions.

            Returns:
                out
            """
            _x = np.asanyarray(x, np.float64)[()]
            return cast(_T_x, _x * 0. + .0)  # :+)

        return influence0

    s, t = clean_trim(trim)

    if l_moment is None:
        lm = l_moment_from_cdf(
            cast(Callable[[float], float], cdf),
            _r,
            trim=(s, t),
            support=support,
            quad_opts=quad_opts,
            alpha=alpha,
        )
    else:
        lm = l_moment

    a, b = support or _tighten_cdf_support(cast(_Fn1, cdf), support)
    c = _l_moment_const(_r, s, t)

    def influence(x: _T_x, /) -> _T_x:
        _x = np.asanyarray(x, np.float64)
        q = np.piecewise(
            _x,
            [_x <= a, (_x > a) & (_x < b), _x >= b],
            [0, cdf, 1],
        )
        w = round0(c * q**s * (1 - q)**t, tol)

        # cheat a bit and replace 0 * inf by 0, ensuring convergence if s or t
        alpha = w * eval_sh_jacobi(_r - 1, t, s, q) * np.where(w, _x, 0)

        return cast(_T_x, round0(alpha - lm, tol)[()])

    influence.__doc__ = (
        f'Theoretical influence function for L-moment with {r=} and {trim=}.'
    )

    return influence


def l_ratio_influence_from_cdf(
    cdf: Callable[[_ArrF8], _ArrF8],
    r: AnyOrder,
    k: AnyOrder = 2,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    l_moments: _Pair[float] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
    tol: float = 1e-8,
) -> Callable[[_T_x], _T_x]:
    r"""
    Construct the influence function of a theoretical L-moment ratio.

    $$
    \psi_{\tau^{(s, t)}_{r,k}|F}(x) = \frac{
        \psi_{\lambda^{(s, t)}_r|F}(x)
        - \tau^{(s, t)}_{r,k} \, \psi_{\lambda^{(s, t)}_k|F}(x)
    }{
        \lambda^{(s,t)}_k
    } \;,
    $$

    where the generalized L-moment ratio is defined as

    $$
    \tau^{(s, t)}_{r,k} = \frac{
        \lambda^{(s, t)}_r
    }{
        \lambda^{(s, t)}_k
    } \;.
    $$

    Because IF's are a special case of the general Gâteuax derivative, the
    L-ratio IF is derived by applying the chain rule to the
    [L-moment IF][lmo.theoretical.l_moment_influence_from_cdf].


    Args:
        cdf: Vectorized cumulative distribution function (CDF).
        r: L-moment ratio order, i.e. the order of the numerator L-moment.
        k: Denominator L-moment order, defaults to 2.
        trim: Left- and right- trim lengths. Defaults to (0, 0).

    Other parameters:
        support:
            The subinterval of the nonzero domain of `cdf`.
        l_moments:
            The L-moments corresponding to $r$ and $k$. If not provided, they
            are calculated from the CDF.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        alpha: Two-sided quantile to split the integral at.
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The influence function, with vectorized signature `() -> ()`.

    See Also:
        - [`l_ratio_from_cdf`][lmo.theoretical.l_ratio_from_cdf]
        - [`lmo.l_ratio`][lmo.l_ratio]

    """
    _r, _k = clean_order(int(r)), clean_order(int(k))

    kwds: dict[str, Any] = {'support': support, 'quad_opts': quad_opts}

    if l_moments is None:
        l_r, l_k = l_moment_from_cdf(
            cast(Callable[[float], float], cdf),
            [_r, _k],
            trim=trim,
            alpha=alpha,
            **kwds,
        )
    else:
        l_r, l_k = l_moments

    if_r = l_moment_influence_from_cdf(
        cdf,
        _r,
        trim,
        l_moment=l_r,
        tol=0,
        **kwds,
    )
    if_k = l_moment_influence_from_cdf(
        cdf,
        _k,
        trim,
        l_moment=l_k,
        tol=0,
        **kwds,
    )

    if abs(l_k) <= tol:
        msg = f'L-ratio ({r=}, {k=}) denominator is approximately zero.'
        raise ZeroDivisionError(msg)
    t_r = l_r / l_k

    def influence_function(x: _T_x, /) -> _T_x:
        psi_r = if_r(x)
        # cheat a bit to avoid `inf - inf = nan` situations
        psi_k = np.where(np.isinf(psi_r), 0, if_k(x))

        return cast(_T_x, round0((psi_r - t_r * psi_k) / l_k, tol=tol)[()])

    influence_function.__doc__ = (
        f'Theoretical influence function for L-moment ratio with r={_r}, '
        f'k={_k}, and {trim=}.'
    )

    return influence_function


# Multivariate

def l_comoment_from_pdf(
    pdf: Callable[[_ArrF8], float],
    cdfs: Sequence[Callable[[float], float]],
    r: AnyOrder,
    /,
    trim: AnyTrim = 0,
    *,
    supports: Sequence[_Pair[float]] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
) -> _ArrF8:
    r"""
    Evaluate the theoretical L-*co*moment matrix of a multivariate probability
    distribution, using the joint PDF
    $f(\vec x) \equiv f(x_1, x_2, \ldots, x_n)$ of random vector $\vec{X}$,
    and the marginal CDFs $F_k$ of its $k$-th random variable.

    The L-*co*moment matrix is defined as

    $$
    \Lambda_{r}^{(s, t)} =
        \left[
            \lambda_{r [ij]}^{(s, t)}
        \right]_{n \times n}
    \;,
    $$

    with elements

    $$
    \begin{align*}
    \lambda_{r [ij]}^{(s, t)}
        &= c^{(s,t)}_r \int_{\mathbb{R^n}}
            x_i \
            u_j^s \
            (1 - u_j)^t \
            \widetilde{P}^{(t, s)}_{r - 1} (u_j) \
            f(\vec{x}) \
            \mathrm{d} \vec{x} \\
        &= c^{(s,t)}_r \, \mathbb{E}_{\vec{X}} \left[
            X_i \
            U_j^s \
            (1 - U_j)^t \
            \widetilde{P}^{(t, s)}_{r - 1}(U_j)
        \right]
        \, ,
    \end{align*}
    $$

    where $U_j = F_j(X_j)$ and $u_j = F_j(x_j)$ denote the (marginal)
    [probability integral transform
    ](https://wikipedia.org/wiki/Probability_integral_transform) of
    $X_j$ and $x_j \sim X_j$.
    Furthermore, $\widetilde{P}^{(\alpha, \beta)}_k$ is a shifted Jacobi
    polynomial, and

    $$
    c^{(s,t)}_r =
        \frac{r + s + t}{r}
        \frac{\B(r,\ r + s + t)}{\B(r + s,\ r + t)} \;
    ,
    $$

    a positive constant.

    For $r \ge 2$, it can also be expressed as

    $$
    \lambda_{r [ij]}^{(s, t)}
        = c^{(s,t)}_r \mathrm{Cov} \left[
            X_i, \;
            U_j^s \
            (1 - U_j)^t \
            \widetilde{P}^{(t, s)}_{r - 1}(U_j)
        \right] \;
        ,
    $$

    and without trim ($s = t = 0$), this simplifies to

    $$
    \lambda_{r [ij]}
        = \mathrm{Cov} \left[
            X_i ,\;
            \widetilde{P}_{r - 1} (U_j)
        \right] \;
        ,
    $$

    with $\tilde{P}_n = \tilde{P}^{(0, 0)}_n$ the shifted Legendre polynomial.
    This last form is precisely the definition introduced by
    Serfling & Xiao (2007).

    Note that the L-comoments along the diagonal, are equivalent to the
    (univariate) L-moments, i.e.

    $$
    \lambda_{r [ii]}^{(s, t)}\big( \vec{X} \big)
        = \lambda_{r}^{(s, t)}\big( X_i \big) \;.
    $$

    Notes:
        At the time of writing, trimmed L-comoments have not been explicitly
        defined in the literature. Instead, the author
        ([@jorenham](https://github.com/jorenham/)) derived it
        by generizing the (untrimmed) L-comoment definition by Serfling &
        Xiao (2007), analogous to the generalization of L-moments
        into TL-moments by Elamir & Seheult (2003).

    Examples:
        Find the L-coscale and TL-coscale matrices of the multivariate
        Student's t distribution with 4 degrees of freedom:

        >>> from scipy.stats import multivariate_t
        >>> df = 4
        >>> loc = np.array([0.5, -0.2])
        >>> cov = np.array([[2.0, 0.3], [0.3, 0.5]])
        >>> X = multivariate_t(loc=loc, shape=cov, df=df)

        >>> from scipy.special import stdtr
        >>> std = np.sqrt(np.diag(cov))
        >>> cdf0 = lambda x: stdtr(df, (x - loc[0]) / std[0])
        >>> cdf1 = lambda x: stdtr(df, (x - loc[1]) / std[1])

        >>> l_cov = l_comoment_from_pdf(X.pdf, (cdf0, cdf1), 2)
        >>> l_cov.round(4)
        array([[1.0413, 0.3124],
               [0.1562, 0.5207]])
        >>> tl_cov = l_comoment_from_pdf(X.pdf, (cdf0, cdf1), 2, trim=1)
        >>> tl_cov.round(4)
        array([[0.4893, 0.1468],
               [0.0734, 0.2447]])

        The (Pearson) correlation coefficient can be recovered in several ways:

        >>> cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])  # "true" correlation
        0.3
        >>> l_cov[0, 1] / l_cov[0, 0]
        0.3
        >>> l_cov[1, 0] / l_cov[1, 1]
        0.3
        >>> tl_cov[0, 1] / tl_cov[0, 0]
        0.3
        >>> tl_cov[1, 0] / tl_cov[1, 1]
        0.3

    Args:
        pdf:
            Joint Probability Distribution Function (PDF), that accepts a
            float vector of size $n$, and returns a scalar in $[0, 1]$.
        cdfs:
            Sequence with $n$ marginal CDF's.
        r:
            Non-negative integer $r$ with the L-moment order.
        trim:
            Left- and right- trim, either as a $(s, t)$ tuple with
            $s, t > -1/2$, or $t$ as alias for $(t, t)$.

    Other parameters:
        supports:
            A sequence with $n$ 2-tuples, corresponding to the marginal
            integration limits. Defaults to $[(-\infty, \infty), \dots]$.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].

    Returns:
        lmbda:
            The population L-*co*moment matrix with shape $n \times n$.

    References:
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [R. Serfling & P. Xiao (2007) - A Contribution to Multivariate
          L-Moments: L-Comoment
          Matrices](https://doi.org/10.1016/j.jmva.2007.01.008)
    """
    n = len(cdfs)
    limits = supports or [_tighten_cdf_support(cdf, None) for cdf in cdfs]

    _r = clean_order(int(r))
    s, t = clean_trim(trim)

    l_r = np.empty((n, n))

    c = _l_moment_const(_r, s, t)

    def integrand(i: int, j: int, *xs: float) -> float:
        x = np.asarray(xs)
        q_j = cdfs[j](x[j])
        p_j = eval_sh_jacobi(_r - 1, t, s, q_j)
        return c * x[i] * q_j**s * (1 - q_j)**t * p_j * pdf(x)

    for i, j in np.ndindex(l_r.shape):
        if i == j:
            l_r[i, j] = l_moment_from_cdf(
                cdfs[i],
                _r,
                trim=(s, t),
                support=limits[i],
                quad_opts=quad_opts,
            )
        else:
            l_r[i, j] = cast(
                float,
                spi.nquad(
                    functools.partial(integrand, i, j),
                    limits,
                    opts=quad_opts,
                )[0],
            ) if _r else 0

    return round0(l_r)


def l_coratio_from_pdf(
    pdf: Callable[[_ArrF8], float],
    cdfs: Sequence[Callable[[float], float]],
    r: AnyOrder,
    r0: AnyOrder = 2,
    /,
    trim: AnyTrim = 0,
    *,
    supports: Sequence[_Pair[float]] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
) -> _ArrF8:
    r"""
    Evaluate the theoretical L-*co*moment ratio matrix of a multivariate
    probability distribution, using the joint PDF $f_{\vec{X}}(\vec{x})$ and
    $n$ marginal CDFs $F_X(x)$ of random vector $\vec{X}$.

    $$
    \tilde \Lambda_{r,r_0}^{(s, t)} =
        \left[
            \left. \lambda_{r [ij]}^{(s, t)} \right/
            \lambda_{r_0 [ii]}^{(s, t)}
        \right]_{n \times n}
    $$

    See Also:
        - [`l_comoment_from_pdf`][lmo.theoretical.l_comoment_from_pdf]
        - [`lmo.l_coratio`][lmo.l_coratio]
    """
    ll_r = l_comoment_from_pdf(
        pdf,
        cdfs,
        r,
        trim=trim,
        supports=supports,
        quad_opts=quad_opts,
    )
    ll_r0 = l_comoment_from_pdf(
        pdf,
        cdfs,
        r0,
        trim=trim,
        supports=supports,
        quad_opts=quad_opts,
    )

    return ll_r / np.expand_dims(ll_r0.diagonal(), -1)


class _VectorizedPPF(Protocol):
    @overload
    def __call__(
        self,
        u: lnpt.AnyArrayInt | lnpt.AnyArrayFloat,
        /,
        *,
        r_max: int = ...,
    ) -> _ArrF8: ...
    @overload
    def __call__(
        self,
        u: lnpt.AnyScalarInt | lnpt.AnyScalarFloat,
        /,
        *,
        r_max: int = ...,
    ) -> float: ...


def _validate_l_bounds(
    l_r: _ArrF8,
    s: float,
    t: float,
) -> None:
    if (l2 := l_r[1]) <= 0:
        msg = f'L-scale must be >0, got lmda[1] = {l2}'
        raise ValueError(msg)

    if len(l_r) <= 2:
        return

    # enforce the (non-strict) L-ratio bounds, from Hosking (2007) eq. 14,
    # but rewritten using falling factorials, to avoid potential overflows
    tau = l_r[2:] / l2

    _r = np.arange(3, len(l_r) + 1)
    m = max(s, t) + 1
    tau_absmax = 2 * fpow(_r + s + t, m) / (_r * fpow(2 + s + t, m))

    if np.any(invalid := np.abs(tau) > tau_absmax):
        r_invalid = list(np.argwhere(invalid) + 3)
        if len(r_invalid) == 1:
            r_invalid = r_invalid[0]
        msg = (
            f'L-moment(s) with r = {r_invalid}) are not within the valid'
            f'range'
        )
        raise ValueError(msg)

    # validate an l-skewness / l-kurtosis relative inequality that is
    # a pre-condition for the PPF to be strictly monotonically increasing
    t3 = tau[0]
    t4 = tau[1] if len(tau) > 1 else 0

    m = 2 + (s if t3 > 0 else t)
    u = 3 + s + t
    t3_max = 2 * (u / m + (m + 1) * (u + 4) * t4) / (3 * (u + 2))

    if abs(t3) >= t3_max:
        if t3 < 0:
            msg_t3_size, msg_t3_trim = 'small', 's'
        else:
            msg_t3_size, msg_t3_trim = 'large', 't'

        msg = (
            f'L-skewness is too {msg_t3_size} ({t3:.4f}); consider '
            f'increasing {msg_t3_trim}'
        )
        raise ValueError(msg)


def _monotonic(
    f: Callable[[_ArrF8], _ArrF8],
    a: float,
    b: float,
    n: int = 100,
    strict: bool = False,
) -> bool:
    """Numeric validation of the monotinicity of a function on [a, b]."""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    # dy = np.gradient(y)
    dy = np.ediff1d(y)

    return bool(np.all(dy > 0 if strict else dy >= 0))


def ppf_from_l_moments(
    lmbda: lnpt.AnyVectorFloat,
    /,
    trim: AnyTrim = 0,
    *,
    support: _Pair[float] = (-np.inf, np.inf),
    validate: bool = True,
    extrapolate: bool = False,
) -> _VectorizedPPF:
    r"""
    Return a PPF (quantile function, or inverse CDF), with the specified.
    L-moments \( \tlmoment{s, t}{1}, \tlmoment{s, t}{2}, \ldots,
    \tlmoment{s, t}{R} \). Other L-moments are considered zero.

    For \( R \) L-moments, this function returns

    \[
        \hat{Q}_R(u) = \sum_{r=1}^{R}
            r \frac{2r + s + t - 1}{r + s + t}
            \tlmoment{s, t}{r}
            \shjacobi{r - 1}{t}{s}{u},
    \]

    where \( \shjacobi{n}{a}{b}{x} \) is an \( n \)-th degree shifted Jacobi
    polynomial, which is orthogonal for \( (a, b) \in (-1, \infty)^2 \) on
    \( u \in [0, 1] \).

    This *nonparametric* quantile function estimation method was first
    described by
    [J.R.M. Hosking in 2007](https://doi.org/10.1016/j.jspi.2006.12.002).
    However, his derivation contains a small, but obvious error, resulting
    in zero-division for \( r = 1 \).
    So Lmo derived this correct version  himself, by using the fact that
    L-moments are the disguised coefficients of the PPF's generalized
    Fourier-Jacobi series expansion.

    With Parseval's theorem it can be shown that, if the probability-weighted
    moment \( M_{2,s,t} \) (which is the variance if \( s = t = 0 \)) is
    finite, then \( \hat{Q}_R(u) = Q(u) \) as \( R \to \infty \).

    Args:
        lmbda:
            1-d array-like of L-moments \( \tlmoment{s,t}{r} \) for
            \( r = 1, 2, \ldots, R \). At least 2 L-moments are required.
            All remaining L-moments with \( r > R \) are considered zero.
        trim:
            The trim-length(s) of L-moments `lmbda`.
        support:
            A tuple like `(x_min, x_max)`. If provided, the PPF results
            will be clipped to within this interval.
        validate:
            If `True` (default), a `ValueError` will be raised if the
            resulting PPF is invalid (non-monotonic), which can be solved by
            increasing  the `trim`.
        extrapolate:
            If set to `True`, a simple moving average of \( R \) and
            \( R - 1 \) will be returned. This generally results in a smoother
            and more accurate PPF, but its L-moments will not be equal to
            `lmda`. Defaults to `False`.

    Returns:
        ppf:
            A vectorized PPF (quantile function). Its extra optional
            keyword argument `r_max: int` can be used to "censor" trailing
            L-moments, i.e. truncating the degree of the polynomial.

    """
    l_r = np.asarray(lmbda)
    if (_n := len(l_r)) < 2:
        msg = f'at least 2 L-moments required, got len(lmbda) = {_n}'
        raise ValueError(msg)

    s, t = clean_trim(trim)

    if validate:
        _validate_l_bounds(l_r, s, t)

    a, b = support
    if a >= b:
        msg = f'invalid support; expected a < b, got a, b = {a}, {b}'
        raise ValueError(msg)

    # r = np.arange(1, _n + 1)
    # c = (2 * r + s + t - 1) * (r / (r + s + t)) * l_r
    w = np.arange(1, 2 * _n + 1, 2, dtype=np.float64)
    if (st := s + t) != 0:
        w -= st * np.arange(_n) / np.arange(st + 1, _n + st + 1)
    c = w * l_r

    def ppf(
        u: npt.ArrayLike,
        *,
        r_max: int = -1,
    ) -> float | _ArrF8:
        y = np.asarray(u)
        y = np.where((y < 0) | (y > 1), np.nan, 2 * y - 1)

        _c = c[:r_max] if 0 < r_max < len(c) else c

        x = fourier_jacobi(y, _c, t, s)
        if extrapolate and _n > 2:
            x = (x + fourier_jacobi(y, _c[:-1], t, s)) / 2

        return np.clip(x, *support)[()]

    if validate and not _monotonic(cast(_VectorizedPPF, ppf), 0, 1):
        msg = (
            'PPF is not monotonically increasing (not invertable); '
            'consider increasing the trim'
        )
        raise ValueError(msg)

    return cast(_VectorizedPPF, ppf)


def qdf_from_l_moments(
    lmbda: lnpt.AnyVectorFloat,
    /,
    trim: AnyTrim = 0,
    *,
    validate: bool = True,
    extrapolate: bool = False,
) -> _VectorizedPPF:
    r"""
    Return the QDF (quantile density function, the derivative of the PPF),
    with the specified L-moments \( \tlmoment{s, t}{1}, \tlmoment{s, t}{2},
    \ldots, \tlmoment{s, t}{R} \). Other L-moments are considered zero.

    This function returns

    \[
    \begin{align*}
        \hat{q}_R(u)
            &= \frac{\dd{\hat{Q}_R(u)}}{\dd{u}}  \\
            &= \sum_{r=2}^{R}
                r (2r + s + t - 1)
                \tlmoment{s, t}{r}
                \shjacobi{r - 2}{t + 1}{s + 1}{u},
    \end{align*}
    \]

    where \( \shjacobi{n}{a}{b}{x} \) is an \( n \)-th degree shifted Jacobi
    polynomial, which is orthogonal for \( (a, b) \in (-1, \infty)^2 \) on
    \( u \in [0, 1] \).

    See [`ppf_from_l_moments`][lmo.theoretical.ppf_from_l_moments] for options.
    """
    l_r = np.asarray(lmbda)
    if (_n := len(l_r)) < 2:
        msg = f'at least 2 L-moments required, got len(lmbda) = {_n}'
        raise ValueError(msg)

    s, t = clean_trim(trim)

    if validate:
        _validate_l_bounds(l_r, s, t)

    # r = np.arange(2, _n + 1)
    # c = (2 * r + s + t - 1) * r * l_r[1:]
    st = s + t
    c = (
        np.arange(1 + st, 2 * _n + st + 1, 2, dtype=np.float64)
        * np.arange(1, _n + 1, dtype=np.float64)
        * l_r
    )[1:]
    alpha, beta = t + 1, s + 1

    def qdf(
        u: onpt.AnyFloatingArray,
        *,
        r_max: int = -1,
    ) -> float | _ArrF8:
        y = np.asanyarray(u, dtype=np.float64)
        # TODO: make this lazy
        y = np.where((y < 0) | (y > 1), np.nan, 2 * y - 1)

        _c = c[:r_max] if 0 < r_max < len(c) else c

        x = fourier_jacobi(y, _c, alpha, beta)
        if extrapolate and _n > 2:
            x = (x + fourier_jacobi(y, _c[:-1], alpha, beta)) / 2

        return x[()]

    if validate and np.any(qdf(plotting_positions(100)) < 0):
        msg = 'QDF is not positive; consider increasing the trim'
        raise ValueError(msg)

    return cast(_VectorizedPPF, qdf)


def cdf_from_ppf(
    ppf: Callable[Concatenate[float, _Tss], float],
    /,
) -> Callable[Concatenate[float, _Tss], float]:
    """Numerical inversion of the PPF."""
    from scipy.optimize import (
        root_scalar,  # pyright: ignore[reportUnknownVariableType]
    )

    def cdf(x: float, /, *args: _Tss.args, **kwds: _Tss.kwargs) -> float:
        if np.isnan(x):
            return np.nan
        if x <= ppf(0, *args, **kwds):
            return 0.
        if x >= ppf(1, *args, **kwds):
            return 1.

        def _ppf_to_solve(p: float) -> float:
            return ppf(p, *args, **kwds) - x

        return cast(
            float,
            root_scalar(_ppf_to_solve, bracket=[0, 1], method='brentq').root,
        )

    return cdf


def entropy_from_qdf(
    qdf: Callable[Concatenate[float, _Tss], float],
    /,
    *args: _Tss.args,
    **kwds: _Tss.kwargs,
) -> float:
    r"""
    Evaluate the (differential / continuous) entropy \( H(X) \) of a
    univariate random variable \( X \), from its *quantile density
    function* (QDF), \( q(u) = \frac{\mathrm{d} F^{-1}(u)}{\mathrm{d} u} \),
    with \( F^{-1} \) the inverse of the CDF, i.e. the PPF / quantile function.

    The derivation follows from the identity \( f(x) = 1 / q(F(x)) \) of PDF
    \( f \), specifically:

    \[
        h(X)
            = \E[-\ln f(X)]
            = \int_\mathbb{R} \ln \frac{1}{f(x)} \mathrm{d} x
            = \int_0^1 \ln q(u) \mathrm{d} u
    \]

    Args:
        qdf ( (float, *Ts, **Ts) -> float):
            The quantile distribution function (QDF).
        *args (*Ts):
            Optional additional positional arguments to pass to `qdf`.
        **kwds (**Ts):
            Optional keyword arguments to pass to `qdf`.

    Returns:
        The differential entropy \( H(X) \).

    See Also:
        - [Differential entropy - Wikipedia
        ](https://wikipedia.org/wiki/Differential_entropy)

    """
    def ic(p: float) -> float:
        return np.log(qdf(p, *args, **kwds))

    return cast(float, spi.quad(ic, 0, 1, limit=QUAD_LIMIT)[0])
