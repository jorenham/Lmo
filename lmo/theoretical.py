"""
Theoretical (population) L-moments of known univariate probability
distributions.
"""

__all__ = (
    'l_moment_from_cdf',
    'l_moment_from_ppf',
    'l_moment_from_rv',

    'l_ratio_from_cdf',
    'l_ratio_from_ppf',
    'l_ratio_from_rv',

    'l_stats_from_cdf',
    'l_stats_from_ppf',
    'l_stats_from_rv',

    'l_moment_cov_from_cdf',
    'l_moment_cov_from_rv',

    'l_stats_cov_from_cdf',
    'l_stats_cov_from_rv',

    'l_moment_influence',
    'l_ratio_influence',
)

import functools
from collections.abc import Callable, Sequence
from math import exp, factorial, gamma, lgamma
from typing import (
    Any,
    Concatenate,
    Final,
    Literal,
    ParamSpec,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
import scipy.integrate as sci  # type: ignore
import scipy.special as scs  # type: ignore
from scipy.stats.distributions import (  # type: ignore
    rv_continuous,
    rv_discrete,
    rv_frozen,
)

from ._distns import rv_method
from ._utils import (
    clean_order,
    clean_orders,
    clean_trim,
    l_stats_orders,
    moments_to_ratio,
    round0,
)
from .typing import AnyFloat, AnyInt, AnyTrim, IntVector, QuadOptions

T = TypeVar('T')
Theta = ParamSpec('Theta')

Pair: TypeAlias = tuple[T, T]

UnivariateCDF: TypeAlias = Callable[[float], float]
UnivariatePPF: TypeAlias = Callable[[float], float]
UnivariateRV: TypeAlias = rv_continuous | rv_discrete | rv_frozen

ALPHA: Final[float] = 0.1
QUAD_LIMIT: Final[int] = 100


def _nquad(
    integrand: Callable[Concatenate[float, float, Theta], float],
    domains: Sequence[Pair[AnyFloat] | Callable[..., Pair[AnyFloat]]],
    opts: QuadOptions | None = None,
    *args: Theta.args,
    **kwds: Theta.kwargs,
) -> float:
    # nquad only has an `args` param for some invalid reason
    fn = functools.partial(integrand, **kwds) if kwds else integrand

    return cast(
        tuple[float, float],
        sci.nquad(fn, domains[::-1], args, opts=opts),
    )[0]


@functools.cache
def _l_moment_const(r: int, s: float, t: float, k: int = 0) -> float:
    if r <= k:
        return 1.0

    # math.lgamma is faster (and has better type annotations) than
    # scipy.special.loggamma.
    if r + s + t <= 20:
        v = gamma(r + s + t + 1) / (gamma(r + s) * gamma(r + t))
    else:
        v = exp(lgamma(r + s + t + 1) - lgamma(r + s) - lgamma(r + t))
    return factorial(r - 1 - k) / r * v


@overload
def _eval_sh_jacobi(n: int, a: float, b: float, x: float) -> float:
    ...


@overload
def _eval_sh_jacobi(
    n: int,
    a: float,
    b: float,
    x: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    ...


def _eval_sh_jacobi(
    n: int,
    a: float,
    b: float,
    x: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """
    Fast evaluation of the n-th shifted Jacobi polynomial.
    Faster than pre-computing using np.Polynomial, and than
    `scipy.special.eval_jacobi` for n < 4.

    Todo:
        move to _poly, vectorize, annotate, document, test

    """
    if n == 0:
        return 1

    u = 2 * x - 1

    if a == b == 0:
        if n == 1:
            return u

        v = x * (x - 1)

        if n == 2:
            return 1 + 6 * v
        if n == 3:
            return (1 + 10 * v) * u
        if n == 4:
            return 1 + 10 * v * (2 + 7 * v)

        return scs.eval_sh_legendre(n, x)

    if n == 1:
        return (a + b + 2) * x - b - 1
    if n == 2:
        return (
            b * (b + 3)
            - (a + b + 3) * (
                2 * b + 4
                - (a + b + 4) * x
            ) * x
        ) / 2 + 1
    if n == 3:
        return (
            (1 + a) * (2 + a) * (3 + a)
            + (4 + a + b) * (
                3 * (2 + a) * (3 + a)
                + (5 + a + b) * (
                    3 * (3 + a)
                    + (6 + a + b) * (x - 1)
                ) * (x - 1)
            ) * (x - 1)
        ) / 6

    # don't use `eval_sh_jacobi`: https://github.com/scipy/scipy/issues/18988
    return scs.eval_jacobi(n, a, b, u)


def _tighten_cdf_support(
    cdf: UnivariateCDF,
    support: Pair[float] | None = None,
) -> Pair[float]:
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


def _rv_melt(
    rv: UnivariateRV,
    *args: float,
    **kwds: float,
) -> tuple[rv_continuous | rv_discrete, float, float, tuple[float, ...]]:
    """
    Extract and validate the loc/scale and shape args from the potentially
    frozen `scipy.stats` distribution.
    Returns the `rv_continuous` distribution, loc, scale and shape args.
    """
    if isinstance(rv, rv_frozen):
        dist, args, kwds = (
            cast(rv_continuous, rv.dist),
            cast(tuple[float, ...], rv.args),
            cast(dict[str, float], rv.kwds),
        )
    else:
        dist = rv

    shapes, loc, scale = cast(
        tuple[tuple[float, ...], float, float],
        dist._parse_args(*args, **kwds),  # type: ignore
    )
    if scale <= 0:
        msg = f'scale must be >0, got {scale}'
        raise ValueError(msg)
    if invalid_args := set(np.argwhere(1 - dist._argcheck(*shapes))):
        invalid_params = {
            cast(str, param.name): args[i]
            for i, param in enumerate(dist._param_info())  # type: ignore
            if i in invalid_args
        }
        invalid_repr = ', '.join(f'{k}={v}' for k, v in invalid_params.items())
        msg = (
            f'shape arguments ({invalid_repr}) of are invalid for '
            f'{dist.name!r}'
        )
        raise ValueError(msg)

    return dist, loc, scale, shapes


# pyright: reportUnknownMemberType=false,reportPrivateUsage=false
def _rv_fn(
    rv: UnivariateRV,
    name: Literal['cdf', 'ppf'],
    transform: bool,
    /,
    *args: float,
    **kwds: float,
) -> tuple[
    Callable[[float], float],
    Pair[float],
    float,
    float,
]:
    """
    Get the unvectorized cdf or ppf from a `scipy.stats` distribution,
    and apply the loc, scale and shape arguments.
    Return the function, its support, the loc, and the scale.
    """
    dist, loc, scale, shapes = _rv_melt(rv, *args, **kwds)
    assert scale > 0

    m_x, s_x = (loc, scale) if transform else (0, 1)

    a0, b0 = cast(tuple[float, float], dist._get_support(*shapes))
    a, b = m_x + s_x * a0, m_x + s_x * b0

    # prefer the unvectorized implementation if exists
    if f'_{name}_single' in type(dist).__dict__:
        fn_raw = cast(Callable[..., float], getattr(dist, f'_{name}_single'))
    else:
        fn_raw = cast(Callable[..., float], getattr(dist, f'_{name}'))

    if name == 'ppf':
        def ppf(q: float, /) -> float:
            if q < 0 or q > 1:
                return np.nan
            if q == 0:
                return a
            if q == 1:
                return b
            return m_x + s_x * fn_raw(q, *shapes)

        fn = ppf
        support = 0, 1
    else:
        def cdf(x: float, /) -> float:
            if x <= a:
                return 0
            if x >= b:
                return 1
            return fn_raw((x - m_x) / s_x, *shapes)

        fn = cdf
        support = a, b

    return fn, support, loc, scale

def _stack_orders(
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
) -> npt.NDArray[np.int_]:
    return np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))


@overload
def l_moment_from_cdf(
    cdf: UnivariateCDF,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] | None = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
    ppf: UnivariatePPF | None = ...,
) -> np.float_:
    ...


@overload
def l_moment_from_cdf(
    cdf: UnivariateCDF,
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] | None = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
    ppf: UnivariatePPF | None = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_cdf(
    cdf: UnivariateCDF,
    r: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] | None = None,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: UnivariatePPF | None = None,
) -> np.float_ | npt.NDArray[np.float_]:
    r"""
    Evaluate the population L-moment of a continuous probability distribution,
    using its Cumulative Distribution Function (CDF) $F_X(x) = P(X \le x)$.

    $$
    \lambda^{(s, t)}_r =
    \begin{cases}
        1 & r = 0 \\
        \int_{-\infty}^{\infty}
            \left(H(x) - I_{F(x)}(s+1, \,t+1)\right)
            \,\mathrm{d} x
        & r = 1 \\
        \frac{c^{(r,s)}_r}{r}
        \int_{-\infty}^{\infty}
            F(x)^{s+1}
            \left(1 - F(x)\right)^{t+1}
            \,\tilde{P}^{(t+1, s+1)}_{r-2}\big(F(x)\big)
            \,\mathrm{d} x
        & r > 1 \;,
    \end{cases}
    $$

    where

    $$
    c^{(r,s)}_r =
    \frac{r+s+t}{r}
    \frac{B(r,\,r+s+t)}{B(r+s,\,r+t)} \;,
    $$

    $\tilde{P}^{(a,b)}_n(x)$ the shifted ($x \mapsto 2x-1$) Jacobi
    polynomial, $H(x)$ the Heaviside step function, and $I_x(\alpha, \beta)$
    the regularized incomplete gamma function.

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

    def integrand(x: float, _r: int) -> float:
        p = cdf(x)
        if _r == 1:
            if s or t:  # noqa: SIM108
                v = cast(float, scs.betainc(s + 1, t + 1, p))  # type: ignore
            else:
                v = p
            return np.heaviside(x, .5) - v

        return (
            p ** (s + 1)
            * (1 - p) ** (t + 1)
            * _eval_sh_jacobi(_r - 2, t + 1, s + 1, p)
        )

    a, d = support or _tighten_cdf_support(cdf, support)
    b, c = (ppf(alpha), ppf(1 - alpha)) if ppf else (a, d)

    kwds = quad_opts or {}
    kwds.setdefault('limit', QUAD_LIMIT)

    def _l_moment_single(_r: int) -> float:
        if _r == 0:
            return 1

        return _l_moment_const(_r, s, t, 1) * cast(
            float,
            (sci.quad(integrand, a, b, (_r,), **kwds)[0] if a < b else 0) +
            sci.quad(integrand, b, c, (_r,), **kwds)[0] +
            (sci.quad(integrand, c, d, (_r,), **kwds)[0] if c < d else 0),
        )

    l_r_cache: dict[int, float] = {}
    l_r = np.empty_like(rs, dtype=np.float_)
    for i, _r in np.ndenumerate(rs):
        _k = int(_r)
        if _k in l_r_cache:
            l_r[i] = l_r_cache[_k]
        else:
            l_r[i] = l_r_cache[_k] = _l_moment_single(_k)

    return round0(l_r)[()]  # convert back to scalar if needed


@overload
def l_moment_from_ppf(
    ppf: UnivariatePPF,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float_:
    ...


@overload
def l_moment_from_ppf(
    ppf: UnivariatePPF,
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_ppf(
    ppf: UnivariatePPF,
    r: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] = (0, 1),
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
) -> np.float_ | npt.NDArray[np.float_]:
    r"""
    Evaluate the population L-moment of a univariate probability distribution,
    using its Percentile Function (PPF) $x(F)$, also commonly known as the
    quantile function, which is the inverse of the Cumulative Distribution
    Function (CDF).

    $$
    \lambda^{(s, t)}_r =
    c^{(r,s)}_r
    \int_0^1
        F^s (1 - F)^t
        \,\tilde{P}^{(t, s)}_{r-1}(F)
        \,x(F)
        \,\mathrm{d} F
    \;,
    $$

    where

    $$
    c^{(r,s)}_r =
    \frac{r+s+t}{r}
    \frac{B(r,\,r+s+t)}{B(r+s,\,r+t)} \;,
    $$

    and $\tilde{P}^{(a,b)}_n(x)$ the shifted ($x \mapsto 2x-1$) Jacobi
    polynomial.

    Notes:
        Numerical integration is performed with
        [`scipy.integrate.quad`][scipy.integrate.quad], which cannot verify
        whether the integral exists and is finite. If it returns an error
        message, an `IntegrationWarning` is issues, and `nan` is returned
        (even if `quad` returned a finite result).

    Examples:
        Evaluate the first 4 L- and TL-moments of the standard normal
        distribution:

        >>> from scipy.special import ndtri  # standard normal inverse CDF
        >>> l_moment_from_ppf(ndtri, [1, 2, 3, 4])
        array([0.        , 0.56418958, 0.        , 0.06917061])
        >>> l_moment_from_ppf(ndtri, [1, 2, 3, 4], trim=1)
        array([0.        , 0.29701138, 0.        , 0.01855727])

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
        return p**s * (1 - p) ** t * _eval_sh_jacobi(_r - 1, t, s, p) * ppf(p)

    quad_kwds = quad_opts or {}
    quad_kwds.setdefault('limit', QUAD_LIMIT)

    def _l_moment_single(_r: int) -> float:
        if _r == 0:
            return 1

        a, b, c, d = support[0], alpha, 1 - alpha, support[1]
        return _l_moment_const(_r, s, t) * cast(
            float,
            sci.quad(integrand, a, b, (_r,), **quad_kwds)[0] +
            sci.quad(integrand, b, c, (_r,), **quad_kwds)[0] +
            sci.quad(integrand, c, d, (_r,), **quad_kwds)[0],
        )

    l_r_cache: dict[int, float] = {}
    l_r = np.empty_like(rs, dtype=np.float_)
    for i, _r in np.ndenumerate(rs):
        _k = int(_r)
        if _k in l_r_cache:
            l_r[i] = l_r_cache[_k]
        else:
            l_r[i] = l_r_cache[_k] = _l_moment_single(_k)

    return round0(l_r)[()]  # convert back to scalar if needed


@overload
def l_moment_from_rv(
    rv: UnivariateRV,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float_:
    ...


@overload
def l_moment_from_rv(
    rv: UnivariateRV,
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_rv(
    rv: UnivariateRV,
    r: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
) -> np.float_ | npt.NDArray[np.float_]:
    r"""
    Evaluate the population L-moment of a univariate
    [`scipy.stats`][scipy.stats] probability distribution.

    $$
    \lambda^{(s, t)}_r =
    \frac{r+s+t}{r}
    \frac{B(r,\,r+s+t)}{B(r+s,\,r+t)}
    \mathbb{E}_X \left[
        U^s
        \left(1 - U\right)^t
        \,\tilde{P}^{(t, s)}_{r-1}(U)
        \,X
    \right] \;,
    $$

    with $U = F_X(X)$ the *rank* of $X$, and $\tilde{P}^{(a,b)}_n(x)$ the
    shifted ($x \mapsto 2x-1$) Jacobi polynomial.

    Examples:
        Evaluate the population L-moments of the normally-distributed IQ test:

        >>> from scipy.stats import norm
        >>> l_moment_from_rv(norm(100, 15), [1, 2, 3, 4]).round(6)
        array([100.      ,   8.462844,   0.      ,   1.037559])
        >>> _[1] * np.sqrt(np.pi)
        15.000000...

        Discrete distributions are also supported, e.g. the Binomial
        distribution:

        >>> from scipy.stats import binom
        >>> l_moment_from_rv(binom(10, .6), [1, 2, 3, 4]).round(6)
        array([ 6.      ,  0.862238, -0.019729,  0.096461])

    Args:
        rv:
            Univariate [`scipy.stats`][scipy.stats] `rv_continuous`,
            `rv_discrete` or `rv_frozen` instance.
        r:
            L-moment order(s), non-negative integer or array-like of integers.
        trim:
            Left- and right- trim. Must be a tuple of two non-negative ints
            or floats.

    Other parameters:
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        alpha:
            Split the integral into integrals with limits $[a, \alpha]$,
            $[\alpha, 1-\alpha]$ and $[1-\alpha, b]$ to improve numerical
            stability. So $\alpha$ can be consideresd the size of the tail.
            Numerical experiments have found 0.1 to give good results for
            different distributions.

    Raises:
        TypeError: `r` is not integer-valued
        ValueError: `r` is empty or negative

    Returns:
        lmbda:
            The population L-moment(s), a scalar or float array like `r`.

    References:
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    See Also:
        - [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf]:
          population L-moment, using the cumulative distribution function.
        - [`l_moment_from_ppf`][lmo.theoretical.l_moment_from_ppf]:
          population L-moment, using the inverse CDF (quantile function).
        - [`lmo.l_moment`][lmo.l_moment]: sample L-moment
    """
    rs = clean_orders(np.asanyarray(r))

    cdf, support, loc, scale = _rv_fn(rv, 'cdf', False)
    ppf = _rv_fn(rv, 'ppf', False)[0]

    assert scale > 0, scale

    lm = l_moment_from_cdf(
        cdf,
        rs,
        trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
        ppf=ppf,
    )
    if loc == 0 and scale == 1:
        return lm

    lms = np.asarray(lm)
    lms[rs == 1] += loc
    lms[rs > 1] *= scale

    return lms[()]  # convert back to scalar if needed

@overload
def l_ratio_from_cdf(
    cdf: UnivariateCDF,
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] | None = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float_:
    ...


@overload
def l_ratio_from_cdf(
    cdf: UnivariateCDF,
    r: IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] | None = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
    ppf: UnivariatePPF | None = ...,
) -> npt.NDArray[np.float_]:
    ...


@overload
def l_ratio_from_cdf(
    cdf: UnivariateCDF,
    r: AnyInt | IntVector,
    s: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] | None = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
    ppf: UnivariatePPF | None = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_ratio_from_cdf(
    cdf: UnivariateCDF,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] | None = None,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: UnivariatePPF | None = None,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    Population L-ratio's from a CDF.

    See Also:
        - [`l_ratio_from_ppf`][lmo.theoretical.l_ratio_from_ppf]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = _stack_orders(r, s)
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
    ppf: UnivariatePPF,
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float_:
    ...


@overload
def l_ratio_from_ppf(
    ppf: UnivariatePPF,
    r: IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> npt.NDArray[np.float_]:
    ...


@overload
def l_ratio_from_ppf(
    ppf: UnivariatePPF,
    r: AnyInt | IntVector,
    s: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: Pair[float] = ...,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_ratio_from_ppf(
    ppf: UnivariatePPF,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] = (0, 1),
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    Population L-ratio's from a PPF.

    See Also:
        - [`l_ratio_from_cdf`][lmo.theoretical.l_ratio_from_cdf]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = _stack_orders(r, s)
    l_rs = l_moment_from_ppf(
        ppf,
        rs,
        trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
    )
    return moments_to_ratio(rs, l_rs)


@overload
def l_ratio_from_rv(
    rv: UnivariateRV,
    r: AnyInt,
    s: AnyInt = ...,
    /,
    trim: AnyTrim = ...,
    *,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> np.float_:
    ...


@overload
def l_ratio_from_rv(
    rv: UnivariateRV,
    r: IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> npt.NDArray[np.float_]:
    ...


@overload
def l_ratio_from_rv(
    rv: UnivariateRV,
    r: AnyInt | IntVector,
    s: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    quad_opts: QuadOptions | None = ...,
    alpha: float = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_ratio_from_rv(
    rv: UnivariateRV,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector = 2,
    /,
    trim: AnyTrim = (0, 0),
    *,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
) -> np.float_ | npt.NDArray[np.float_]:
    r"""
    Population L-ratio's from a [`scipy.stats`][scipy.stats] univariate
    continuous probability distribution.

    See [`l_moment_from_rv`][lmo.theoretical.l_moment_from_rv] for a
    description of the parameters.

    Examples:
        Evaluate the population L-CV and LL-CV (CV = coefficient of variation)
        of the standard Rayleigh distribution.

        >>> from scipy.stats import distributions
        >>> X = distributions.rayleigh()
        >>> X.std() / X.mean()
        0.5227232...
        >>> l_ratio_from_rv(X, 2, 1)
        0.2928932...
        >>> l_ratio_from_rv(X, 2, 1, trim=(0, 1))
        0.2752551...

        And similarly, for the (discrete) Poisson distribution with rate
        parameter set to 2, the L-CF and LL-CV evaluate to:

        >>> X = distributions.poisson(2)
        >>> X.std() / X.mean()
        0.7071067...
        >>> l_ratio_from_rv(X, 2, 1)
        0.3857527...
        >>> l_ratio_from_rv(X, 2, 1, trim=(0, 1))
        0.4097538...

        Note that (untrimmed) L-CV requires a higher (subdivision) limit in
        the integration routine, otherwise it'll complain that it didn't
        converge (enough) yet. This is because it's effectively integrating
        a non-smooth function, which is mathematically iffy, but works fine
        in this numerical application.

    See Also:
        - [`l_moment_from_rv`][lmo.theoretical.l_moment_from_rv]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = _stack_orders(r, s)
    l_rs = l_moment_from_rv(
        rv,
        rs,
        trim,
        quad_opts=quad_opts,
        alpha=alpha,
    )
    return moments_to_ratio(rs, l_rs)


def l_stats_from_cdf(
    cdf: UnivariateCDF,
    num: int = 4,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] | None = None,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: UnivariatePPF | None = None,
) -> npt.NDArray[np.float_]:
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
    ppf: UnivariatePPF,
    num: int = 4,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] = (0, 1),
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
) -> npt.NDArray[np.float_]:
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


def l_stats_from_rv(
    rv: UnivariateRV,
    num: int = 4,
    /,
    trim: AnyTrim = (0, 0),
    *,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
) -> npt.NDArray[np.float_]:
    r"""
    Calculates the theoretical- / population- L-moments (for $r \le 2$)
    and L-ratio's (for $r > 2$) of a [`scipy.stats`][scipy.stats] distribution.

    By default, the first `num = 4` population L-stats are calculated:

    - $\lambda^{(s,t)}_1$ - *L-loc*ation
    - $\lambda^{(s,t)}_2$ - *L-scale*
    - $\tau^{(s,t)}_3$ - *L-skew*ness coefficient
    - $\tau^{(s,t)}_4$ - *L-kurt*osis coefficient

    This function is equivalent to
    `l_ratio_from_rv(rv, [1, 2, 3, 4], [0, 0, 2, 2], *, **)`.

    See [`l_moment_from_rv`][lmo.theoretical.l_moment_from_rv] for a
    description of the parameters.

    Examples:
        Summarize the standard exponential distribution for different trims.

        >>> from scipy.stats import distributions
        >>> X = distributions.expon()
        >>> l_stats_from_rv(X).round(6)
        array([1.      , 0.5     , 0.333333, 0.166667])
        >>> l_stats_from_rv(X, trim=(0, 1/2)).round(6)
        array([0.666667, 0.333333, 0.266667, 0.114286])
        >>> l_stats_from_rv(X, trim=(0, 1)).round(6)
        array([0.5     , 0.25    , 0.222222, 0.083333])

    Note:
        This should not be confused with the term *L-statistic*, which is
        sometimes used to describe any linear combination of order statistics.

    See Also:
        - [`l_ratio_from_rv`][lmo.theoretical.l_ratio_from_ppf]
        - [`lmo.l_stats`][lmo.l_stats] - Unbiased sample estimation of L-stats.
    """
    return l_ratio_from_rv(
        rv,
        *l_stats_orders(num),
        trim,
        quad_opts=quad_opts,
        alpha=alpha,
    )


def l_moment_cov_from_cdf(
    cdf: UnivariateCDF,
    r_max: int,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] | None = None,
    quad_opts: QuadOptions | None = None,
) -> npt.NDArray[np.float_]:
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
    \begin{align}
    \bf{\Lambda}^{(s,t)}_{k, r}
        &= \mathrm{Cov}[l^{(s, t)}_k, l^{(s, t)}_r] \\
        &= c_k c_r
        \iint\limits_{x < y} \Big[
            p_k\big(F(x)\big) \, p_r\big(F(y)\big) +
            p_r\big(F(x)\big) \, p_k\big(F(y)\big)
        \Big]
        w^{(s+1,\, t)}\big(F(x)\big) \,
        w^{(s,\, t+1)}\big(F(y)\big) \,
        \mathrm{d}x \, \mathrm{d}y
        \;,
    \end{align}
    $$

    where

    $$
    c_n = \frac{\Gamma(n) \Gamma(n+s+t+1)}{n \Gamma(n+s) \Gamma(n+t)}\;,
    $$

    the shifted Jacobi polynomial
    $p_n(u) = P^{(t, s)}_{n-1}(2u - 1)$, $P^{(t, s)}_m$, and
    $w^{(s,t)}(u) = u^s (1-u)^t$ its weight function.

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
        return  c_n[k] * c_n[r] * (
            (
                _eval_sh_jacobi(k, t, s, u)
                * _eval_sh_jacobi(r, t, s, v)
                + _eval_sh_jacobi(r, t, s, u)
                * _eval_sh_jacobi(k, t, s, v)
            )
            * u * (1 - v)
            * (u * v)**s * ((1 - u) * (1 - v))**t
        )

    def range_x(y: float, *_: int) -> tuple[float, float]:
        return (a, y)

    cov = np.empty((rs, rs), dtype=np.float_)
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


def l_moment_cov_from_rv(
    rv: UnivariateRV,
    r_max: int,
    /,
    trim: AnyTrim = (0, 0),
    *,
    quad_opts: QuadOptions | None = None,
) -> npt.NDArray[np.float_]:
    """
    Calculate the asymptotic L-moment covariance matrix from a
    [`scipy.stats`][scipy.stats] distribution.

    See [`l_moment_cov_from_cdf`][lmo.theoretical.l_moment_cov_from_cdf] for
    more info.

    Examples:
        >>> from scipy.stats import distributions
        >>> X = distributions.expon()  # standard exponential distribution
        >>> l_moment_cov_from_rv(X, 4).round(6)
        array([[1.      , 0.5     , 0.166667, 0.083333],
               [0.5     , 0.333333, 0.166667, 0.083333],
               [0.166667, 0.166667, 0.133333, 0.083333],
               [0.083333, 0.083333, 0.083333, 0.071429]])

        >>> l_moment_cov_from_rv(X, 4, trim=(0, 1)).round(6)
        array([[0.333333, 0.125   , 0.      , 0.      ],
               [0.125   , 0.075   , 0.016667, 0.      ],
               [0.      , 0.016667, 0.016931, 0.00496 ],
               [0.      , 0.      , 0.00496 , 0.0062  ]])

    """
    cdf, support, _, scale = _rv_fn(rv, 'cdf', False)
    cov = l_moment_cov_from_cdf(
        cdf,
        r_max,
        trim,
        support=support,
        quad_opts=quad_opts,
    )
    return scale**2 * cov


def l_stats_cov_from_cdf(
    cdf: UnivariateCDF,
    num: int = 4,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] | None = None,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
    ppf: UnivariatePPF | None = None,
) -> npt.NDArray[np.float_]:
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

    l_r0 = l_moment_from_cdf(
        cdf,
        np.arange(2, rs + 1),
        trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
        ppf=ppf,
    )

    l_2 = l_r0[0]
    assert l_2 > 0, l_2

    t_r = np.r_[np.nan, l_r0 / l_2]

    cov = np.empty_like(ll_kr)
    for k, r in zip(*np.triu_indices(rs), strict=True):
        assert k <= r, (k, r)
        assert ll_kr[k, r] == ll_kr[r, k]

        if r <= 1:
            tt = ll_kr[k, r]
        elif k <= 1:
            tt = (ll_kr[k, r] - ll_kr[1, k] * t_r[r]) / l_2
        else:
            tt = (
                ll_kr[k, r]
                - ll_kr[1, k] * t_r[r]
                - ll_kr[1, r] * t_r[k]
                + ll_kr[1, 1] * t_r[k] * t_r[r]
            ) / l_2**2

        cov[k, r] = cov[r, k] = tt

    return round0(cov)


def l_stats_cov_from_rv(
    rv: UnivariateRV,
    num: int = 4,
    /,
    trim: AnyTrim = (0, 0),
    *,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
) -> npt.NDArray[np.float_]:
    """
    Calculate the asymptotic L-stats covariance matrix from a
    [`scipy.stats`][scipy.stats] distribution.

    See [`l_stats_cov_from_cdf`][lmo.theoretical.l_stats_cov_from_cdf] for
    more info.

    Examples:
        Evaluate the LL-stats covariance matrix of the standard exponential
        distribution, for 0, 1, and 2 degrees of trimming.

        >>> from scipy.stats import distributions
        >>> X = distributions.expon()  # standard exponential distribution
        >>> l_stats_cov_from_rv(X).round(6)
        array([[1.      , 0.5     , 0.      , 0.      ],
               [0.5     , 0.333333, 0.111111, 0.055556],
               [0.      , 0.111111, 0.237037, 0.185185],
               [0.      , 0.055556, 0.185185, 0.21164 ]])
        >>> l_stats_cov_from_rv(X, trim=(0, 1)).round(6)
        array([[ 0.333333,  0.125   , -0.111111, -0.041667],
               [ 0.125   ,  0.075   ,  0.      , -0.025   ],
               [-0.111111,  0.      ,  0.21164 ,  0.079365],
               [-0.041667, -0.025   ,  0.079365,  0.10754 ]])
        >>> l_stats_cov_from_rv(X, trim=(0, 2)).round(6)
        array([[ 0.2     ,  0.066667, -0.114286, -0.02    ],
               [ 0.066667,  0.038095, -0.014286, -0.023333],
               [-0.114286, -0.014286,  0.228571,  0.04    ],
               [-0.02    , -0.023333,  0.04    ,  0.086545]])

        Note that with 0 trim the L-location is independent of the
        L-skewness and L-kurtosis. With 1 trim, the L-scale and L-skewness
        are independent. And with 2 trim, all L-stats depend on each other.

    """
    cdf, support, _, scale = _rv_fn(rv, 'cdf', False)
    ppf = _rv_fn(rv, 'ppf', False)[0]

    cov = l_stats_cov_from_cdf(
        cdf,
        num,
        trim,
        support=support,
        quad_opts=quad_opts,
        alpha=alpha,
        ppf=ppf,
    )
    if scale != 1 and num:
        cov[:2, :2] *= scale**2

    return cov


def l_moment_influence(
    rv_or_cdf: (
        UnivariateRV
        | Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]
    ),
    r: SupportsIndex,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] | None = None,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
    tol: float = 1e-8,
) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float_]]:
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
        rv_or_cdf:
            Either a [`scipy.stats`][scipy.stats] continuous distribution
            instance, or a (vectorized) cumulative distribution function (CDF).
        r: The L-moment order. Must be a non-negative integer.
        trim: Left- and right- trim lengths. Defaults to (0, 0).

    Other parameters:
        support:
            The subinterval of the nonzero domain of `cdf`. This is ignored if
            a `scipy.stats` distribution is used.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        alpha: Two-sided quantile to split the integral at.
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The influence function, with vectorized signature `() -> ()`.

    See Also:
        - [`l_ratio_influence`][lmo.theoretical.l_ratio_influence]
        - [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf]
        - [`l_moment_from_rv`][lmo.theoretical.l_moment_from_rv]
        - [`lmo.l_moment`][lmo.l_moment]
    """
    _r = clean_order(r)
    if _r == 0:
        def influence_function(
            x: npt.ArrayLike,
            /,
        ) -> float | npt.NDArray[np.float_]:
            """
            L-moment Influence Function for `r=0`.

            Args:
                x: Scalar or array-like of sample observarions.

            Returns:
                out
            """
            return np.asarray(x, np.float_) * 0. + .0  # :+)

        return influence_function

    s, t = clean_trim(trim)

    if isinstance(rv_or_cdf, UnivariateRV):
        lm = l_moment_from_rv(
            rv_or_cdf,
            _r,
            trim,
            quad_opts=quad_opts,
            alpha=alpha,
        )
        cdf = cast(
            Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
            rv_or_cdf.cdf,
        )
    else:
        lm = l_moment_from_cdf(
            cast(Callable[[float], float], rv_or_cdf),
            _r,
            trim,
            support=support,
            quad_opts=quad_opts,
        )
        cdf = rv_or_cdf

    c = _l_moment_const(_r, s, t)

    def influence_function(
        x: npt.ArrayLike,
        /,
    ) -> float | npt.NDArray[np.float_]:
        _x = np.asanyarray(x, np.float_)
        q = cdf(_x)
        w = round0(c * q**s * (1 - q)**t, tol)

        # cheat a bit and replace 0 * inf by 0, ensuring convergence if s or t
        alpha = w * _eval_sh_jacobi(_r - 1, t, s, q) * np.where(w, _x, 0)

        return round0(alpha - lm, tol)[()]

    influence_function.__doc__ = (
        f'Theoretical influence function for L-moment with {r=} and {trim=}.'
    )

    return influence_function


def l_ratio_influence(
    rv_or_cdf: (
        UnivariateRV
        | Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]
    ),
    r: SupportsIndex,
    k: SupportsIndex = 2,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: Pair[float] | None = None,
    quad_opts: QuadOptions | None = None,
    alpha: float = ALPHA,
    tol: float = 1e-8,
) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float_]]:
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
    [L-moment IF][lmo.theoretical.l_moment_influence].


    Args:
        rv_or_cdf:
            Either a [`scipy.stats`][scipy.stats] continuous distribution
            instance, or a (vectorized) cumulative distribution function (CDF).
        r: L-moment ratio order, i.e. the order of the numerator L-moment.
        k: Denominator L-moment order, defaults to 2.
        trim: Left- and right- trim lengths. Defaults to (0, 0).

    Other parameters:
        support:
            The subinterval of the nonzero domain of `cdf`. This is ignored if
            a `scipy.stats` distribution is used.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        alpha: Two-sided quantile to split the integral at.
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The influence function, with vectorized signature `() -> ()`.

    See Also:
        - [`l_moment_influence`][lmo.theoretical.l_moment_influence]
        - [`l_ratio_from_cdf`][lmo.theoretical.l_ratio_from_cdf]
        - [`l_ratio_from_rv`][lmo.theoretical.l_ratio_from_rv]
        - [`lmo.l_ratio`][lmo.l_ratio]

    """
    _r, _k = clean_order(r), clean_order(k)

    kwds: dict[str, Any] = {'support': support, 'quad_opts': quad_opts}
    if_r = l_moment_influence(rv_or_cdf, r, trim, tol=0, **kwds)
    if_k = l_moment_influence(rv_or_cdf, k, trim, tol=0, **kwds)

    if isinstance(rv_or_cdf, UnivariateRV):
        tau_r, lambda_k = l_ratio_from_rv(
            rv_or_cdf,
            [_r, _k],
            [_k, 0],
            trim=trim,
            quad_opts=quad_opts,
            alpha=alpha,
        )
    else:
        tau_r, lambda_k = l_ratio_from_cdf(
            cast(Callable[[float], float], rv_or_cdf),
            [_r, _k],
            [_k, 0],
            trim=trim,
            **kwds,
        )

    def influence_function(
        x: npt.ArrayLike,
        /,
    ) -> float | npt.NDArray[np.float_]:
        psi_r = if_r(x)
        # cheat a bit to avoid `inf - inf = nan` situations
        psi_k = np.where(np.isinf(psi_r), 0, if_k(x))

        return round0((psi_r - tau_r * psi_k) / lambda_k, tol=tol)

    influence_function.__doc__ = (
        f'Theoretical influence function for L-moment ratio with r={_r}, '
        f'k={_k}, and {trim=}.'
    )

    return influence_function


"""
Methods to be added to `scipy.stats.rv_generic` and `scipy.stats.rv_frozen`.
"""


@rv_method('l_moment')
def _rv_l_moment(  # type: ignore
    self: rv_continuous | rv_discrete,
    order: AnyInt | IntVector,
    /,
    *args: float,
    trim: AnyTrim = (0, 0),
    quad_opts: QuadOptions | None = None,
    **kwds: float,
) -> np.float_ | npt.NDArray[np.float_]:
    """L-moment(s) of distribution of specified order(s).

    Parameters
    ----------
    order : array_like
        Order(s) of L-moment(s).
    arg1, arg2, arg3,... : float
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : float, optional
        location parameter (default=0)
    scale : float, optional
        scale parameter (default=1)
    trim : float or tuple, optional
        left- and right- trim (default=(0, 0))

    Returns
    -------
    lm : ndarray or scalar
        The calculated L-moment(s).

    """  # noqa: D416
    rs = clean_orders(np.asanyarray(order))

    args, loc, scale = cast(
        tuple[tuple[float, ...], float, float],
        self._parse_args(*args, **kwds),  # type: ignore
    )
    support = cast(tuple[float, float], self._get_support(*args))

    _cdf = cast(Callable[[float], float], self._cdf)
    _ppf = cast(Callable[[float], float], self._ppf)

    if args:
        def cdf(x: float, /) -> float:
            return _cdf(x, *args)

        def ppf(q: float, /):
            return _ppf(q, *args)
    else:
        cdf, ppf = _cdf, _ppf

    lm = np.asarray(l_moment_from_cdf(
        cdf,
        rs,
        trim=trim,
        support=support,
        ppf=ppf,
        quad_opts=quad_opts,
    ))
    lm[rs == 1] += loc
    lm[rs > 1] *= scale
    return lm[()]  # convert back to scalar if needed


@rv_method('l_ratio')
def _rv_l_ratio(  # type: ignore
    self: rv_continuous | rv_discrete,
    order: AnyInt | IntVector,
    order_denom: AnyInt | IntVector,
    /,
    *args: float,
    trim: AnyTrim = (0, 0),
    quad_opts: QuadOptions | None = None,
    **kwds: float,
) -> np.float_ | npt.NDArray[np.float_]:
    """L-moment ratio('s) of distribution of specified order(s).

    Parameters
    ----------
    order : array_like
        Order(s) of L-moment(s).
    order_denom : array_like
        Order(s) of L-moment denominator(s).
    arg1, arg2, arg3,... : float
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : float, optional
        location parameter (default=0)
    scale : float, optional
        scale parameter (default=1)
    trim : float or tuple, optional
        left- and right- trim (default=(0, 0))

    Returns
    -------
    tm : ndarray or scalar
        The calculated L-moment ratio('s).

    """  # noqa: D416
    rs = _stack_orders(order, order_denom)
    lms = cast(
        npt.NDArray[np.float_],
        self.l_moment(  # type: ignore
            rs,
            *args,
            trim=trim,
            quad_opts=quad_opts,
            **kwds,
        ),
    )
    return moments_to_ratio(rs, lms)


@rv_method('l_stats')
def _rv_l_stats(  # type: ignore
    self: rv_continuous | rv_discrete,
    *args: float,
    trim: AnyTrim = (0, 0),
    moments: int = 4,
    quad_opts: QuadOptions | None = None,
    **kwds: float,
) -> np.float_ | npt.NDArray[np.float_]:
    """L-moments (order <= 2) and L-moment ratio's (order > 2).

    By default, the first `num = 4` L-stats are calculated. This is
    equivalent to `l_ratio([1, 2, 3, 4], [0, 0, 2, 2], *, **)`, i.e. the
    L-location, L-scale, L-skew, and L-kurtosis.

    Parameters
    ----------
    arg1, arg2, arg3,... : float
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : float, optional
        location parameter (default=0)
    scale : float, optional
        scale parameter (default=1)
    trim : float or tuple, optional
        left- and right- trim (default=(0, 0))
    moments : int, optional
        the amount of L-moment stats to compute (default=4)

    Returns
    -------
    tm : ndarray or scalar
        The calculated L-moment ratio('s).

    """  # noqa: D416
    r, s = l_stats_orders(moments)
    return cast(
        npt.NDArray[np.float_],
        self.l_ratio(  # type: ignore
            r,
            s,
            *args,
            trim=trim,
            quad_opts=quad_opts,
            **kwds,
        ),
    )


@rv_method('l_loc')
def _rv_l_loc(  # type: ignore
    self: rv_continuous | rv_discrete,
    *args: float,
    trim: AnyTrim = (0, 0),
    **kwds: float,
) -> float:
    """L-location of the distribution, i.e. the 1st L-moment.

    Without trim (default), the L-location is equivalent to the mean.

    Parameters
    ----------
    arg1, arg2, arg3,... : float
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : float, optional
        location parameter (default=0)
    scale : float, optional
        scale parameter (default=1)
    trim : float or tuple, optional
        left- and right- trim (default=(0, 0))

    Returns
    -------
    l_loc : float
        The L-location of the distribution.

    """  # noqa: D416
    if not any(clean_trim(trim)):
        return cast(float, self.mean(*args, **kwds))

    return cast(
        float,
        self.l_moment(1, *args, trim=trim, **kwds),  # type: ignore
    )


@rv_method('l_scale')
def _rv_l_scale(  # type: ignore
    self: rv_continuous | rv_discrete,
    *args: float,
    trim: AnyTrim = (0, 0),
    **kwds: float,
) -> float:
    """L-scale of the distribution, i.e. the 2nd L-moment.

    Without trim (default), the L-location is equivalent to half the Gini
    mean (absolute) difference (GMD).

    Just like the standard deviation, the L-scale is location-invariant, and
    varies proportionally to positive scaling.

    Parameters
    ----------
    arg1, arg2, arg3,... : float
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : float, optional
        location parameter (default=0)
    scale : float, optional
        scale parameter (default=1)
    trim : float or tuple, optional
        left- and right- trim (default=(0, 0))

    Returns
    -------
    l_scale : float
            The L-scale of the distribution.

    """  # noqa: D416
    return cast(
        float,
        self.l_moment(2, *args, trim=trim, **kwds),  # type: ignore
    )


@rv_method('l_skew')
def _rv_l_skew(  # type: ignore
    self: rv_continuous | rv_discrete,
    *args: float,
    trim: AnyTrim = (0, 0),
    **kwds: float,
) -> float:
    """L-skewness coefficient of the distribution; the 3rd L-moment ratio.

    Parameters
    ----------
    arg1, arg2, arg3,... : float
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : float, optional
        location parameter (default=0)
    scale : float, optional
        scale parameter (default=1)
    trim : float or tuple, optional
        left- and right- trim (default=(0, 0))

    Returns
    -------
    l_skew : float
        The L-skewness coefficient of the distribution.

    """  # noqa: D416
    return cast(
        float,
        self.l_ratio(3, 2, *args, trim=trim, **kwds),  # type: ignore
    )


@rv_method('l_kurtosis')
def _rv_l_kurtosis(  # type: ignore
    self: rv_continuous | rv_discrete,
    *args: float,
    trim: AnyTrim = (0, 0),
    **kwds: float,
) -> float:
    """L-kurtosis coefficient of the distribution; the 4th L-moment ratio.

    Parameters
    ----------
    arg1, arg2, arg3,... : float
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : float, optional
        location parameter (default=0)
    scale : float, optional
        scale parameter (default=1)
    trim : float or tuple, optional
        left- and right- trim (default=(0, 0))

    Returns
    -------
    l_kurtosis : float
        The L-kurtosis coefficient of the distribution.

    """  # noqa: D416
    return cast(
        float,
        self.l_ratio(4, 2, *args, trim=trim, **kwds),  # type: ignore
    )
