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
)

import functools
import warnings
from collections.abc import Callable, Sequence
from math import exp, lgamma, log
from typing import Any, Final, Literal, cast, overload

import numpy as np
import numpy.typing as npt
import scipy.integrate as sci  # type: ignore
import scipy.special as scs  # type: ignore
from scipy.stats.distributions import rv_continuous, rv_frozen  # type: ignore

from ._utils import (
    clean_order,
    clean_trim,
    l_stats_orders,
    moments_to_ratio,
    round0,
)
from .linalg import sh_jacobi
from .typing import AnyFloat, AnyInt, AnyTrim, IntVector

DEFAULT_RTOL: Final[float] = 1.49e-8
DEFAULT_ATOL: Final[float] = 1.49e-8
DEFAULT_LIMIT: Final[int] = 50


def _l_moment_const(r: int, s: float, t: float, k: int = 0) -> float:
    if r <= k:
        return 1.0

    # math.lgamma is faster (and has better type annotations) than
    # scipy.special.loggamma.
    return exp(
        lgamma(r - k)
        + lgamma(r + s + t + 1)
        - lgamma(r + s)
        - lgamma(r + t)
        - log(r),
    )


def _tighten_cdf_support(
    cdf: Callable[[float], float],
    support: tuple[AnyFloat, AnyFloat],
) -> tuple[float, float]:
    """Attempt to tighten the support by checking some common bounds."""
    a, b = map(float, support)

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


def _quad(
    integrand: Callable[[float], float],
    domain: tuple[AnyFloat, AnyFloat],
    limit: int,
    atol: float,
    rtol: float,
) -> float:
    quad_val, _, _, *quad_tail = sci.quad(  # type: ignore
        integrand,
        *domain,
        full_output=True,
        limit=limit,
        epsabs=atol,
        epsrel=rtol,
    )
    if quad_tail:
        msg = f"'scipy.integrate.quad' failed: \n{quad_tail[0]}"
        warnings.warn(msg, sci.IntegrationWarning, stacklevel=2)
        return np.nan

    return cast(float, quad_val)


def _nquad(
    integrand: Callable[..., float],
    domains: Sequence[
        tuple[AnyFloat, AnyFloat] | Callable[..., tuple[float, float]]
    ],
    limit: int,
    atol: float,
    rtol: float,
    args: tuple[Any, ...] = (),
) -> float:
    quad_val, _ = cast(
        tuple[float, float],
        sci.nquad(  # type: ignore
            integrand,
            domains[::-1],
            args,
            opts={'limit': limit, 'epsabs': atol, 'epsrel': rtol},
        ),
    )
    return quad_val


@overload
def l_moment_from_cdf(
    cdf: Callable[[float], float],
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
) -> np.float_:
    ...


@overload
def l_moment_from_cdf(
    cdf: Callable[[float], float],
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_cdf(  # noqa: C901
    cdf: Callable[[float], float],
    r: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: tuple[AnyFloat, AnyFloat] = (-np.inf, np.inf),
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
) -> np.float_ | npt.NDArray[np.float_]:
    r"""
    Evaluate the population L-moment of a continuous probability distribution,
    using its Cumulative Distribution Function (CDF) $F_X(x) = P(X \le x)$.

    Notes:
        Numerical integration is performed with
        [`scipy.integrate.quad`][scipy.integrate.quad], which cannot verify
        whether the integral exists and is finite. If it returns an error
        message, an `IntegrationWarning` is issues, and `nan` is returned
        (even if `quad` returned a finite result).

    Args:
        cdf:
            Cumulative Distribution Function (CDF), $F_X(x) = P(X \le x)$.
            Must be a continuous monotone increasing function with
            signature `(float) -> float`, whose return value lies in $[0, 1]$.
        r:
            L-moment order(s), non-negative integer or array-like of integers.
        trim:
            Left- and right- trim. Must be a tuple of two non-negative ints
            or floats (!).

    Other parameters:
        support: The subinterval of the nonzero domain of `cdf`.
            Generally it's not needed to provide this, as it will be guessed
            automatically.
        rtol: See `epsrel` [`scipy.integrate.quad`][scipy.integrate.quad].
        atol: See `epsabs` [`scipy.integrate.quad`][scipy.integrate.quad].
        limit: See `limit` in [`scipy.integrate.quad`][scipy.integrate.quad].

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
    _r = np.asanyarray(r)
    if not np.issubdtype(_r.dtype, np.integer):
        msg = 'r must be integer-valued, got {_r.dtype.str!r}'
        raise TypeError(msg)
    if np.any(_r < 0):
        msg = 'r must be non-negative'
        raise TypeError(msg)

    if _r.size == 0:
        return np.empty(_r.shape)

    r_vals, r_idxs = np.unique(_r, return_inverse=True)

    s, t = clean_trim(trim)
    trimmed = s != 0 or t != 0

    a, b = _tighten_cdf_support(cdf, support)

    j = sh_jacobi(min(12, r_vals[-1]) - 1, t + 1, s + 1)

    # caching F(x) function only makes sense for multiple quad calls
    _cdf = functools.cache(cdf) if np.count_nonzero(r_vals) > 1 else cdf

    l_r = np.empty(r_vals.shape)
    for i, r_val in np.ndenumerate(r_vals):
        if r_val == 0:
            # zeroth l-moment is always 1
            l_r[i] = 1
            continue

        if r_val == 1:

            def integrand(x: float, *args: Any) -> float:
                # equivalent to E[X_{s+1 : s+t+1}]
                # see Wiley (2003) eq. 2.1.5
                i_p = p = _cdf(x, *args)
                if trimmed:
                    i_p = scs.betainc(s + 1, t + 1, p)  # type: ignore

                return (x >= 0) - i_p

        else:
            k_val = r_val - 2

            if r_val <= 12:
                c_k, lb = j[k_val, : k_val + 1], 0
            else:
                _j_k = scs.jacobi(k_val, t + 1, s + 1)  # type: ignore
                c_k, lb = _j_k.coef[::-1], -1

            j_k = np.polynomial.Polynomial(c_k, domain=[0, 1], window=[lb, 1])

            # avoid overflows: split in sign and log, and recombine later
            # j_k_sgn = np.sign(j_k)
            # j_k_ln = np.log(np.abs(j_k))

            def integrand(x: float, *args: Any) -> float:
                """
                Evaluate the jacobi polynomial for p at r-1 with (t, s)
                and multiply by the weight function.
                """
                p = _cdf(x, *args)
                return (
                    p ** (s + 1) * (1 - p) ** (t + 1) * j_k(p)  # type: ignore
                )

        quad_val = _quad(integrand, (a, b), limit, atol, rtol)
        l_r[i] = _l_moment_const(r_val, s, t, 1) * quad_val

    return (np.round(l_r, 12) + 0.0)[r_idxs].reshape(_r.shape)[()]


@overload
def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
) -> np.float_:
    ...


@overload
def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_ppf(
    ppf: Callable[[float], float],
    r: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: tuple[AnyFloat, AnyFloat] = (0, 1),
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    Evaluate the population L-moment of a continuous probability distribution,
    using its Percentile Function (PPF) $Q_X(p) = F^{-1}_X(p)$,
    i.e. the inverse of the CDF, commonly known as the quantile function.

    Notes:
        Numerical integration is performed with
        [`scipy.integrate.quad`][scipy.integrate.quad], which cannot verify
        whether the integral exists and is finite. If it returns an error
        message, an `IntegrationWarning` is issues, and `nan` is returned
        (even if `quad` returned a finite result).

    Args:
        ppf:
            The quantile function, a monotonically continuous increasing
            function with signature `(float) -> float`, that maps a
            probability in $[0, 1]$, to the domain of the distribution.
        r:
            L-moment order(s), non-negative integer or array-like of integers.
        trim:
            Left- and right- trim. Must be a tuple of two non-negative ints
            or floats (!).

    Other parameters:
        support: The subinterval of the nonzero domain of `cdf`.
            Generally it's not needed to provide this, as it will be guessed
            automatically.
        rtol: See `epsrel` [`scipy.integrate.quad`][scipy.integrate.quad].
        atol: See `epsabs` [`scipy.integrate.quad`][scipy.integrate.quad].
        limit: See `limit` in [`scipy.integrate.quad`][scipy.integrate.quad].

    Raises:
        TypeError: `r` is not integer-valued
        ValueError: `r` is empty or negative

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
    _r = np.asanyarray(r)
    if not np.issubdtype(_r.dtype, np.integer):
        msg = 'r must be integer-valued, got {_r.dtype.str!r}'
        raise TypeError(msg)
    if np.any(_r < 0):
        msg = 'r must be non-negative'
        raise TypeError(msg)

    if _r.size == 0:
        return np.empty(_r.shape, np.float_)

    r_vals, r_idxs = np.unique(_r, return_inverse=True)
    s, t = clean_trim(trim)

    j = sh_jacobi(min(r_vals[-1], 12), t, s)

    def w(p: float) -> float:
        return p**s * (1 - p) ** t * ppf(p)

    # caching the weight function only makes sense for multiple quad calls
    _w = functools.cache(w) if len(r_vals) > 1 else w

    # lazy import (don't worry; python imports are cached)
    from scipy.special import jacobi  # type: ignore

    l_r = np.empty(r_vals.shape)
    for i, r_val in np.ndenumerate(r_vals):
        if r_val == 0:
            # zeroth l-moment is always 1
            l_r[i] = 1
            continue

        if r_val <= 12:
            j_k = np.polynomial.Polynomial(
                j[r_val - 1, :r_val],
                domain=[0, 1],
                window=[0, 1],
            )
        else:
            j_k = np.polynomial.Polynomial(
                jacobi(r_val - 1, t, s).coef[::-1],  # type: ignore
                domain=[0, 1],
                window=[-1, 1],
            )

        def integrand(p: float) -> float:
            return _w(p) * j_k(p)  # type: ignore

        quad_val = _quad(integrand, support, limit, atol, rtol)
        l_r[i] = _l_moment_const(r_val, s, t, 0) * quad_val

    return (np.round(l_r, 12) + 0.0)[r_idxs].reshape(_r.shape)[()]


def _rv_melt(
    rv: rv_continuous | rv_frozen,
    *args: float,
    **kwds: float,
) -> tuple[rv_continuous, float, float, tuple[float, ...]]:
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
    if invalid_args := set(np.argwhere(1 - dist._argcheck(shapes))):
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
    rv: rv_continuous | rv_frozen,
    name: Literal['cdf', 'ppf'],
    transform: bool,
    /,
    *args: float,
    **kwds: float,
) -> tuple[Callable[[float], float], tuple[float, float], float, float]:
    """
    Get the unvectorized cdf or ppf from a `scipy.stats` distribution,
    and apply the loc, scale and shape arguments.
    Return the function, its support, the loc, and the scale.
    """
    dist, loc, scale, shapes = _rv_melt(rv, *args, **kwds)

    m_x, s_x = (loc, scale) if transform else (0, 1)

    a0, b0 = cast(tuple[float, float], dist._get_support(*shapes))
    a, b = m_x + s_x * a0, m_x + s_x * b0

    # prefer the unvectorized implementation if exists
    if f'_{name}_single' in type(dist).__dict__:
        fn_raw = cast(Callable[..., float], getattr(dist, f'_{name}_single'))
    else:
        fn_raw = cast(Callable[..., float], getattr(dist, f'_{name}'))

    if name == 'ppf':
        def fn(q: float, /) -> float:
            if q < 0 or q > 1:
                return np.nan
            if q == 0:
                return a
            if q == 1:
                return b
            return m_x + s_x * fn_raw(q, *shapes)

        support = 0, 1
    else:
        def fn(x: float, /) -> float:
            if x <= a:
                return 0
            if x >= b:
                return 1
            return fn_raw((x - m_x) / s_x, *shapes)

        support = a, b

    return fn, support, loc, scale


# pyright: reportUnknownMemberType=false,reportPrivateUsage=false
def _rv_fn_select(
    rv: rv_continuous | rv_frozen,
    transform: bool,
    *args: float,
    **kwds: float,
) -> tuple[
    Literal['cdf', 'ppf'],
    Callable[[float], float],
    tuple[float, float],
    float,
    float,
]:
    """
    Select and extract either the CDF of PPF, depending on which one has been
    natively implemented.
    """
    dist = cast(rv_continuous, rv.dist) if isinstance(rv, rv_frozen) else rv

    dist_methods = type(dist).__dict__
    has_cdf = '_cdf_single' in dist_methods or '_cdf' in dist_methods
    has_ppf = '_ppf_single' in dist_methods or '_ppf' in dist_methods

    momtype = dist.moment_type
    if not has_ppf and has_cdf:
        name = 'cdf'
    elif not has_cdf and has_ppf:  # noqa: SIM114
        name = 'ppf'
    elif momtype == 1:
        name = 'ppf'
    else:
        name = 'cdf'

    return name, *_rv_fn(rv, name, True, *args, **kwds)


@overload
def l_moment_from_rv(
    ppf: rv_continuous | rv_frozen,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *rv_args: float,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
    **rv_kwds: float,
) -> np.float_:
    ...


@overload
def l_moment_from_rv(
    ppf: rv_continuous | rv_frozen,
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *rv_args: float,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
    **rv_kwds: float,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_rv(
    rv: rv_continuous | rv_frozen,
    r: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *rv_args: float,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
    **rv_kwds: float,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    Evaluate the population L-moment of a
    [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] probability
    distribution instance or frozen instance.

    Examples:
        Evaluate the population L-moments of the normally-distributed IQ test.

        >>> from scipy.stats import norm
        >>> l_moment_from_rv(norm(100, 15), [1, 2, 3, 4]).round(6)
        array([100.      ,   8.462844,   0.      ,   1.037559])
        >>> _[1] * np.sqrt(np.pi)
        15.000000...

    Notes:
        If you care about performance, it is generally faster to use
        [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf] or
        [`l_moment_from_ppf`][lmo.theoretical.l_moment_from_ppf] directly.
        For instance by using [`scipy.special.ndtr`][scipy.special.ndtr] or
        [`scipy.special.ndtri`][scipy.special.ndtri], instead of
        [`scipy.stats.norm`][scipy.stats.norm].

    Args:
        rv:
            Univariate continuously distributed [`scipy.stats`][scipy.stats]
            random variable (RV).
            Can be generic or grozen, e.g. `scipy.stats.norm` and
            `scipy.stats.norm()` are both allowed.
        r:
            L-moment order(s), non-negative integer or array-like of integers.
        trim:
            Left- and right- trim. Must be a tuple of two non-negative ints
            or floats (!).
        *rv_args:
            Optional positional arguments for the
            [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] instance.
            These are ignored if `rv` is frozen.
        **rv_kwds:
            Optional keyword arguments for the
            [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] instance.
            These are ignored if `rv` is frozen.

    Other parameters:
        rtol: See `epsrel` [`scipy.integrate.quad`][scipy.integrate.quad].
        atol: See `epsabs` [`scipy.integrate.quad`][scipy.integrate.quad].
        limit: See `limit` in [`scipy.integrate.quad`][scipy.integrate.quad].

    Raises:
        TypeError: `r` is not integer-valued
        ValueError: `r` is empty or negative

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
        - [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf]:
          population L-moment, using the cumulative distribution function.
        - [`l_moment_from_ppf`][lmo.theoretical.l_moment_from_ppf]:
          population L-moment, using the inverse CDF (quantile function).
        - [`lmo.l_moment`][lmo.l_moment]: sample L-moment

    """
    fn_name, rv_fn, ab, _, _ = _rv_fn_select(rv, True, *rv_args, **rv_kwds)
    lm_fn = {'cdf': l_moment_from_cdf, 'ppf': l_moment_from_ppf}[fn_name]

    return lm_fn(
        rv_fn,
        r,
        trim,
        support=ab,
        rtol=rtol,
        atol=atol,
        limit=limit,
    )


def _stack_orders(
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
) -> npt.NDArray[np.int_]:
    return np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(s)))


@overload
def l_ratio_from_cdf(
    cdf: Callable[[float], float],
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    **kwargs: Any,
) -> np.float_:
    ...


@overload
def l_ratio_from_cdf(
    cdf: Callable[[float], float],
    r: IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    ...


@overload
def l_ratio_from_cdf(
    cdf: Callable[[float], float],
    r: AnyInt | IntVector,
    s: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    ...


def l_ratio_from_cdf(
    cdf: Callable[[float], float],
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: tuple[AnyFloat, AnyFloat] = (-np.inf, np.inf),
    **kwargs: Any,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    Population L-ratio's from a CDF.

    See Also:
        - [`l_ratio_from_ppf`][lmo.theoretical.l_ratio_from_ppf]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = _stack_orders(r, s)
    l_rs = l_moment_from_cdf(cdf, rs, trim, support=support, **kwargs)

    return moments_to_ratio(rs, l_rs)


@overload
def l_ratio_from_ppf(
    ppf: Callable[[float], float],
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    **kwargs: Any,
) -> np.float_:
    ...


@overload
def l_ratio_from_ppf(
    ppf: Callable[[float], float],
    r: IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    ...


@overload
def l_ratio_from_ppf(
    ppf: Callable[[float], float],
    r: AnyInt | IntVector,
    s: IntVector,
    /,
    trim: AnyTrim = ...,
    *,
    support: tuple[AnyFloat, AnyFloat] = ...,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    ...


def l_ratio_from_ppf(
    ppf: Callable[[float], float],
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: tuple[AnyFloat, AnyFloat] = (0, 1),
    **kwargs: Any,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    Population L-ratio's from a PPF.

    See Also:
        - [`l_ratio_from_cdf`][lmo.theoretical.l_ratio_from_cdf]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = _stack_orders(r, s)
    l_rs = l_moment_from_ppf(ppf, rs, trim, support=support, **kwargs)

    return moments_to_ratio(rs, l_rs)


@overload
def l_ratio_from_rv(
    rv: rv_continuous | rv_frozen,
    r: AnyInt,
    s: AnyInt,
    /,
    trim: AnyTrim = ...,
    *rv_args: float,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
    **rv_kwds: float,
) -> np.float_:
    ...


@overload
def l_ratio_from_rv(
    rv: rv_continuous | rv_frozen,
    r: IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = ...,
    *rv_args: float,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
    **rv_kwds: float,
) -> npt.NDArray[np.float_]:
    ...


@overload
def l_ratio_from_rv(
    rv: rv_continuous | rv_frozen,
    r: AnyInt | IntVector,
    s: IntVector,
    /,
    trim: AnyTrim = ...,
    *rv_args: float,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
    **rv_kwds: float,
) -> npt.NDArray[np.float_]:
    ...


def l_ratio_from_rv(
    rv: rv_continuous | rv_frozen,
    r: AnyInt | IntVector,
    s: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *rv_args: float,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
    **rv_kwds: float,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    Population L-ratio's from a [`scipy.stats`][scipy.stats] univariate
    continuous probability distribution.

    See [`l_moment_from_rv`][lmo.theoretical.l_moment_from_rv] for a
    description of the parameters.

    Examples:
        Evaluate the population L-CV and LH-CV (CV = coefficient of variation)
        of the standard Rayleigh distribution.

        >>> from scipy.stats import distributions
        >>> X = distributions.rayleigh()
        >>> l_ratio_from_rv(X, 2, 1)
        0.2928932...
        >>> l_ratio_from_rv(X, 2, 1, trim=(0, 1))
        0.2752551...

    See Also:
        - [`l_moment_from_rv`][lmo.theoretical.l_moment_from_rv]
        - [`lmo.l_ratio`][lmo.l_ratio]
    """
    rs = _stack_orders(r, s)
    l_rs = l_moment_from_rv(
        rv,
        rs,
        trim,
        *rv_args,
        rtol=rtol,
        atol=atol,
        limit=limit,
        **rv_kwds,
    )

    return moments_to_ratio(rs, l_rs)

def l_stats_from_cdf(
    cdf: Callable[[float], float],
    /,
    num: int = 4,
    trim: AnyTrim = (0, 0),
    *,
    support: tuple[AnyFloat, AnyFloat] = (-np.inf, np.inf),
    **kwargs: Any,
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
    r, s = l_stats_orders(num)
    return l_ratio_from_cdf(cdf, r, s, trim, support=support, **kwargs)


def l_stats_from_ppf(
    ppf: Callable[[float], float],
    /,
    num: int = 4,
    trim: AnyTrim = (0, 0),
    *,
    support: tuple[AnyFloat, AnyFloat] = (0, 1),
    **kwargs: Any,
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
    r, s = l_stats_orders(num)
    return l_ratio_from_ppf(ppf, r, s, trim, support=support, **kwargs)


def l_stats_from_rv(
    rv: rv_continuous | rv_frozen,
    /,
    num: int = 4,
    trim: AnyTrim = (0, 0),
    *rv_args: float,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
    **rv_kwds: float,
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
    r, s = l_stats_orders(num)
    return l_ratio_from_rv(
        rv,
        r,
        s,
        trim,
        *rv_args,
        rtol=rtol,
        atol=atol,
        limit=limit,
        **rv_kwds,
    )


def _eval_sh_jacobi(
    n: int,
    a: float,
    b: float,
    x: float,
) -> float:
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



def l_moment_cov_from_cdf(
    cdf: Callable[[float], float],
    r_max: int,
    /,
    trim: AnyTrim = (0, 0),
    *,
    support: tuple[AnyFloat, AnyFloat] | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
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
    and \vec{\lambda}^{(s, t)} its theoretical ("true") counterpart.

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
        support: The subinterval of the nonzero domain of `cdf`.
            Generally it's not needed to provide this, as it will be guessed
            automatically.
        rtol: See `epsrel` in [`scipy.integrate.nquad`][scipy.integrate.nquad].
        atol: See `epsabs` in [`scipy.integrate.nquad`][scipy.integrate.nquad].
        limit: See `limit` in [`scipy.integrate.nquad`][scipy.integrate.nquad].

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
            limit=limit,
            atol=atol,
            rtol=rtol,
            args=(k, r),
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

    return round0(cov, atol)


def l_moment_cov_from_rv(
    rv: rv_continuous | rv_frozen,
    r_max: int,
    /,
    trim: AnyTrim = (0, 0),
    *rv_args: float,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
    **rv_kwds: float,
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
    cdf, support, _, scale = _rv_fn(
        rv,
        'cdf',
        False,
        *rv_args,
        **rv_kwds,
    )
    return scale**2 * l_moment_cov_from_cdf(
        cdf,
        r_max,
        trim=trim,
        support=support,
        rtol=rtol,
        atol=atol,
        limit=limit,
    )


def l_stats_cov_from_cdf(
    cdf: Callable[[float], float],
    /,
    num: int = 4,
    trim: AnyTrim = (0, 0),
    **kwargs: Any,
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
        **kwargs:
            Optional keyword arguments to pass to
            [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf].

    References:
        - [J.R.M. Hosking (1990) - L-moments: Analysis and Estimation of
            Distributions Using Linear Combinations of Order Statistics
            ](https://jstor.org/stable/2345653)
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
    """
    rs = clean_order(num, 'num', 0)
    ll_kr = l_moment_cov_from_cdf(cdf, rs, trim=trim, **kwargs)
    if rs <= 2:
        return ll_kr

    l_r0 = l_moment_from_cdf(cdf, np.arange(2, rs + 1), trim=trim, **kwargs)

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

    return round0(cov, kwargs.get('atol', DEFAULT_ATOL))


def l_stats_cov_from_rv(
    rv: rv_continuous | rv_frozen,
    /,
    num: int = 4,
    trim: AnyTrim = (0, 0),
    *rv_args: float,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
    **rv_kwds: float,
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
    cdf, support, _, scale = _rv_fn(
        rv,
        'cdf',
        False,
        *rv_args,
        **rv_kwds,
    )
    cov = l_stats_cov_from_cdf(
        cdf,
        num,
        trim=trim,
        support=support,
        rtol=rtol,
        atol=atol,
        limit=limit,
    )
    if scale != 1:
        cov[:2, :2] *= scale**2

    return cov
