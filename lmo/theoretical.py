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
    'l_stats_from_cdf',
    'l_stats_from_ppf',
    'l_moment_cov_from_cdf',
    'l_stats_cov_from_cdf',
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

from . import _poly
from ._utils import clean_order, clean_trim, moments_to_ratio
from .linalg import sh_jacobi
from .typing import AnyFloat, AnyInt, AnyTrim, IntVector

DEFAULT_RTOL: Final[float] = 1.49e-8
DEFAULT_ATOL: Final[float] = 1.49e-8
DEFAULT_LIMIT: Final[int] = 100


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


def _round_like(x: float, tol: float) -> float:
    return round(x, -int(np.log10(tol))) + 0.0 if tol else x


def _quad(
    integrand: Callable[[float], float],
    domain: tuple[AnyFloat, AnyFloat],
    limit: int,
    atol: float,
    rtol: float,
) -> float:
    quad_val, quad_err, _, *quad_tail = sci.quad(  # type: ignore
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

    return _round_like(cast(float, quad_val), cast(float, quad_err))


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
    quad_val, quad_err = cast(
        tuple[float, float],
        sci.nquad(  # type: ignore
            integrand,
            domains[::-1],
            args,
            opts={'limit': limit, 'epsabs': atol, 'epsrel': rtol},
        ),
    )
    return _round_like(quad_val, quad_err)


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
        return np.empty(_r.shape)

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


# pyright: reportUnknownMemberType=false
def _rv_unwrap(
    rv: rv_continuous | rv_frozen,
    *rv_args: Any,
    **rv_kwds: Any,
) -> tuple[
    Literal['cdf', 'ppf'],
    Callable[[float], float],
    tuple[float, float],
]:
    _rv = rv if isinstance(rv, rv_frozen) else rv(*rv_args, *rv_kwds)
    momtype = cast(int, _rv.dist.moment_type)

    if momtype == 0:
        return (
            'cdf',
            cast(Callable[[float], float], _rv.cdf),
            cast(tuple[float, float], _rv.support()),
        )
    if momtype == 1:
        return (
            'ppf',
            cast(Callable[[float], float], _rv.ppf),
            (0, 1),
        )

    msg = f'unknown momtype {momtype!r}'
    raise TypeError(msg)

@overload
def l_moment_from_rv(
    ppf: rv_continuous | rv_frozen,
    r: AnyInt,
    /,
    trim: AnyTrim = ...,
    *rv_args: Any,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
    **rv_kwds: Any,
) -> np.float_:
    ...


@overload
def l_moment_from_rv(
    ppf: rv_continuous | rv_frozen,
    r: IntVector,
    /,
    trim: AnyTrim = ...,
    *rv_args: Any,
    rtol: float = ...,
    atol: float = ...,
    limit: int = ...,
    **rv_kwds: Any,
) -> npt.NDArray[np.float_]:
    ...


def l_moment_from_rv(
    rv: rv_continuous | rv_frozen,
    r: AnyInt | IntVector,
    /,
    trim: AnyTrim = (0, 0),
    *rv_args: Any,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    limit: int = DEFAULT_LIMIT,
    **rv_kwds: Any,
) -> np.float_ | npt.NDArray[np.float_]:
    """
    Evaluate the population L-moment of a
    [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] probability
    distribution instance or frozen instance.

    Examples:
        Evaluate the population L-moments of the normally-distributed IQ test.

        >>> from scipy.stats import distributions
        >>> l_moment_from_rv(distributions.norm(100, 15), [1, 2, 3, 4])
        array([100.       ,   8.4628438,   0.       ,   1.0375592])
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
    _rv = rv if isinstance(rv, rv_frozen) else rv(*rv_args, *rv_kwds)

    functype, func, support = _rv_unwrap(rv, *rv_args, **rv_kwds)

    l_moment_from_func = {
        'cdf': l_moment_from_cdf,
        'ppf': l_moment_from_ppf,
    }[functype]

    return l_moment_from_func(
        func,
        r,
        trim=trim,
        support=support,
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
    r, s = np.arange(1, num + 1), [0] * min(2, num) + [2] * (num - 2)
    return l_ratio_from_cdf(cdf, r, s, trim=trim, support=support, **kwargs)


def l_stats_from_ppf(
    ppf: Callable[[float], float],
    /,
    trim: AnyTrim = (0, 0),
    num: int = 4,
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
    r, s = np.arange(1, num + 1), [0] * min(2, num) + [2] * (num - 2)
    return l_ratio_from_ppf(ppf, r, s, trim=trim, support=support, **kwargs)


def l_moment_cov_from_cdf(
    cdf: Callable[[float], float],
    r_max: int,
    /,
    trim: AnyTrim = (0, 0),
    *,
    scale: float = 1.0,
    support: tuple[AnyFloat, AnyFloat] = (-np.inf, np.inf),
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
        scale:
            The scale of the distribution, defaults to 1. The resulting
            covariances will be divided by `scale**2`.

    Other parameters:
        support: The subinterval of the nonzero domain of `cdf`.
            Generally it's not needed to provide this, as it will be guessed
            automatically.
        rtol: See `epsrel` in [`scipy.integrate.nquad`][scipy.integrate.nquad].
        atol: See `epsabs` in [`scipy.integrate.nquad`][scipy.integrate.nquad].
        limit: See `limit` in [`scipy.integrate.nquad`][scipy.integrate.nquad].

    Returns:
        out: Covariance matrix, with shape `(r_max, r_max)`.

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

    _cdf = functools.cache(cdf)

    a, b = _tighten_cdf_support(_cdf, support)
    s, t = clean_trim(trim)

    p_n = [_poly.jacobi(n, t, s, domain=[0, 1]) for n in range(rs)]
    c_n = np.array([_l_moment_const(n + 1, s, t) for n in range(rs)]) / scale

    def integrand(x: float, y: float, k: int, r: int) -> float:
        u, v = _cdf(x), _cdf(y)
        return  c_n[k] * c_n[r] * cast(
            float,
            (p_n[k](u) * p_n[r](v) + p_n[r](u) * p_n[k](v))
            * u * (u * v)**s
            * (1 - v) * ((1 - u) * (1 - v))**t,
        )

    def range_x(y: float, *_: int) -> tuple[float, float]:
        return (a, y)

    out = np.empty((rs, rs), dtype=np.float_)
    for k, r in zip(*np.triu_indices(rs), strict=True):
        out[k, r] = out[r, k] = _nquad(
            integrand,
            [(a, b), range_x],
            limit=limit,
            atol=atol,
            rtol=rtol,
            args=(k, r),
        )

    return out


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

    out = np.empty_like(ll_kr)
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

        out[k, r] = out[r, k] = tt

    return out
