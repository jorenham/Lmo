from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, TypeAlias, TypeVar, Unpack, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from lmo._poly import eval_sh_jacobi
from lmo._utils import (
    clean_orders,
    clean_trim,
    l_stats_orders,
    moments_to_ratio,
    round0,
)
from ._utils import l_const, tighten_cdf_support

if TYPE_CHECKING:
    from collections.abc import Callable

    import lmo.typing as lmt


__all__ = [
    "l_moment_from_cdf", "l_moment_from_ppf", "l_moment_from_qdf",
    "l_ratio_from_cdf", "l_ratio_from_ppf",
    "l_stats_from_cdf", "l_stats_from_ppf",
]  # fmt: skip


_T = TypeVar("_T")
_Pair: TypeAlias = tuple[_T, _T]

_Float1D: TypeAlias = onp.Array1D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]


ALPHA: Final[float] = 0.1
QUAD_LIMIT: Final[int] = 100


def _df_quad3(
    f: Callable[[float, int], float | npc.floating],
    /,
    a: onp.ToFloat,
    b: onp.ToFloat,
    c: onp.ToFloat,
    d: onp.ToFloat,
    r: int,
    **kwds: Unpack[lmt.QuadOptions],
) -> float:
    import scipy.integrate as spi

    out = spi.quad(f, b, c, (r,), **kwds)[0]
    if a < b:
        out += spi.quad(f, a, b, (r,), **kwds)[0]
    if c < d:
        out += spi.quad(f, c, d, (r,), **kwds)[0]

    return out


@overload
def l_moment_from_cdf(
    cdf: Callable[[float], onp.ToFloat],
    r: lmt.ToOrder0D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: Callable[[float], onp.ToFloat] | None = None,
) -> float: ...
@overload
def l_moment_from_cdf(
    cdf: Callable[[float], onp.ToFloat],
    r: lmt.ToOrderND,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: Callable[[float], onp.ToFloat] | None = None,
) -> _FloatND: ...
def l_moment_from_cdf(
    cdf: Callable[[float], onp.ToFloat],
    r: lmt.ToOrder,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: Callable[[float], onp.ToFloat] | None = None,
) -> float | _FloatND:
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

    def ig(x: float, r_: int) -> float:
        p = np.asarray(cdf(x), np.float64).item()
        if r_ == 1:
            if s or t:  # noqa: SIM108
                v = betainc(s + 1, t + 1, p)
            else:
                v = p
            return np.heaviside(x, 0.5) - v

        return (
            math.sqrt(2 * r_ - 1)
            * p ** (s + 1)
            * (1 - p) ** (t + 1)
            * eval_sh_jacobi(r_ - 2, t + 1, s + 1, p)
        )

    a, d = support or tighten_cdf_support(cdf, support)
    b, c = (ppf(alpha), ppf(1 - alpha)) if ppf else (a, d)

    loc0 = a if np.isfinite(a) and a > 0 else 0

    kwds = quad_opts or {}
    _ = kwds.setdefault("limit", QUAD_LIMIT)

    def _l_moment_single(r_: int) -> float:
        if r_ == 0:
            return 1

        return (
            l_const(r_, s, t, 1) / np.sqrt(2 * r_ - 1)
            * _df_quad3(ig, a, b, c, d, r_, **kwds)
            + loc0 * (r_ == 1)
        )  # fmt: skip

    l_r_cache: dict[int, float] = {}
    l_r = np.empty_like(rs, dtype=np.float64)
    for i, _r in np.ndenumerate(rs):
        k = int(_r)
        if k in l_r_cache:
            l_r[i] = l_r_cache[k]
        else:
            l_r[i] = l_r_cache[k] = _l_moment_single(k)

    return round0(l_r, 1e-12)[()]  # convert back to scalar if needed


@overload
def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: lmt.ToOrder0D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
) -> float: ...
@overload
def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: lmt.ToOrderND,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
) -> _FloatND: ...
def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: lmt.ToOrder,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
) -> float | _FloatND:
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

    def igf(p: float, r_: int, /) -> float:
        return p**s * (1 - p) ** t * eval_sh_jacobi(r_ - 1, t, s, p) * ppf(p)

    kwds = quad_opts or {}
    _ = kwds.setdefault("limit", QUAD_LIMIT)

    def _l_moment_single(r_: int) -> float:
        if r_ == 0:
            return 1

        a, b, c, d = support[0], alpha, 1 - alpha, support[1]
        return l_const(r_, s, t) * _df_quad3(igf, a, b, c, d, r_, **kwds)

    l_r_cache: dict[int, float] = {}
    l_r = np.empty_like(rs, dtype=np.float64)
    for i, _r in np.ndenumerate(rs):
        k = int(_r)
        if k in l_r_cache:
            l_r[i] = l_r_cache[k]
        else:
            l_r[i] = l_r_cache[k] = _l_moment_single(k)

    return round0(l_r.item() if l_r.ndim == 0 and np.isscalar(rs) else l_r)


@overload
def l_moment_from_qdf(
    qdf: Callable[[float], float],
    r: lmt.ToOrder0D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
) -> float: ...
@overload
def l_moment_from_qdf(
    qdf: Callable[[float], float],
    r: lmt.ToOrderND,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
) -> _FloatND: ...
def l_moment_from_qdf(
    qdf: Callable[[float], float],
    r: lmt.ToOrder,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
) -> float | _FloatND:
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
    out *= c_scale

    return out.item() if out.ndim == 0 and np.isscalar(r) else out


@overload
def l_ratio_from_cdf(
    cdf: Callable[[float], float],
    r: lmt.ToOrder0D,
    s: lmt.ToOrder0D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
    ppf: Callable[[float], float] | None = None,
) -> float: ...
@overload
def l_ratio_from_cdf(
    cdf: Callable[[float], float],
    r: lmt.ToOrderND,
    s: lmt.ToOrder,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
    ppf: Callable[[float], float] | None = None,
) -> _FloatND: ...
@overload
def l_ratio_from_cdf(
    cdf: Callable[[float], float],
    r: lmt.ToOrder,
    s: lmt.ToOrderND,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
    ppf: Callable[[float], float] | None = None,
) -> _FloatND: ...
def l_ratio_from_cdf(
    cdf: Callable[[float], float],
    r: lmt.ToOrder,
    s: lmt.ToOrder,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: Callable[[float], float] | None = None,
) -> float | _FloatND:
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
    ppf: Callable[[float], float],
    r: lmt.ToOrder0D,
    s: lmt.ToOrder0D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = ...,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
) -> float: ...
@overload
def l_ratio_from_ppf(
    ppf: Callable[[float], float],
    r: lmt.ToOrderND,
    s: lmt.ToOrder,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = ...,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
) -> _FloatND: ...
@overload
def l_ratio_from_ppf(
    ppf: Callable[[float], float],
    r: lmt.ToOrder,
    s: lmt.ToOrderND,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = ...,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ...,
) -> _FloatND: ...
def l_ratio_from_ppf(
    ppf: Callable[[float], float],
    r: lmt.ToOrder,
    s: lmt.ToOrder,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
) -> float | _FloatND:
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
    cdf: Callable[[float], float],
    num: int = 4,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: Callable[[float], float] | None = None,
) -> _Float1D:
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
    ppf: Callable[[float], float],
    num: int = 4,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (0, 1),
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
) -> _Float1D:
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
