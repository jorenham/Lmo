from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from lmo._utils import clean_trim, plotting_positions
from lmo.special import fourier_jacobi, fpow

if TYPE_CHECKING:
    from collections.abc import Callable

    import optype.numpy as onp

    import lmo.typing as lmt

__all__ = ["ppf_from_l_moments", "qdf_from_l_moments"]

_T = TypeVar("_T")
_T_x = TypeVar("_T_x", float, npt.NDArray[np.float64])

_Pair: TypeAlias = tuple[_T, _T]
_FloatND: TypeAlias = npt.NDArray[np.float64]


class _Fn1(Protocol):
    def __call__(self, x: _T_x, /) -> _T_x: ...


def _validate_l_bounds(l_r: _FloatND, s: float, t: float) -> None:
    if (l2 := l_r[1]) <= 0:
        msg = f"L-scale must be >0, got lmda[1] = {l2}"
        raise ValueError(msg)

    if len(l_r) <= 2:
        return

    # enforce the (non-strict) L-ratio bounds, from Hosking (2007) eq. 14,
    # but rewritten using falling factorials, to avoid potential overflows
    tau = l_r[2:] / l2

    r = np.arange(3, len(l_r) + 1)
    m = max(s, t) + 1
    tau_absmax = 2 * fpow(r + s + t, m) / (r * fpow(2 + s + t, m))

    if np.any(invalid := np.abs(tau) > tau_absmax):
        r_invalid = list(np.argwhere(invalid) + 3)
        if len(r_invalid) == 1:
            r_invalid = r_invalid[0]
        msg = f"L-moment(s) with r = {r_invalid}) are not within the valid range"
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
            msg_t3_size, msg_t3_trim = "small", "s"
        else:
            msg_t3_size, msg_t3_trim = "large", "t"

        msg = (
            f"L-skewness is too {msg_t3_size} ({t3:.4f}); consider "
            f"increasing {msg_t3_trim}"
        )
        raise ValueError(msg)


def _monotonic(
    f: Callable[[_FloatND], np.float64 | _FloatND],
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
    lmbda: onp.ToFloat1D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] = (-np.inf, np.inf),
    validate: bool = True,
    extrapolate: bool = False,
) -> _Fn1:
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
        msg = f"at least 2 L-moments required, got len(lmbda) = {_n}"
        raise ValueError(msg)

    s, t = clean_trim(trim)

    if validate:
        _validate_l_bounds(l_r, s, t)

    a, b = support
    if a >= b:
        msg = f"invalid support; expected a < b, got a, b = {a}, {b}"
        raise ValueError(msg)

    # r = np.arange(1, _n + 1)
    # c = (2 * r + s + t - 1) * (r / (r + s + t)) * l_r
    w = np.arange(1, 2 * _n + 1, 2, dtype=np.float64)
    if (st := s + t) != 0:
        w -= st * np.arange(_n) / np.arange(st + 1, _n + st + 1)
    c = w * l_r

    def ppf(
        u: _T_x,
        *,
        r_max: int = -1,
    ) -> _T_x:
        y = np.asarray(u)
        y = np.where((y < 0) | (y > 1), np.nan, 2 * y - 1)

        c_ = c[:r_max] if 0 < r_max < len(c) else c

        x = fourier_jacobi(y, c_, t, s)
        if extrapolate and _n > 2:
            x = (x + fourier_jacobi(y, c_[:-1], t, s)) / 2

        return np.clip(x, *support)[()]  # pyright: ignore[reportReturnType]

    if validate and not _monotonic(ppf, 0, 1):
        msg = (
            "PPF is not monotonically increasing (not invertable); "
            "consider increasing the trim"
        )
        raise ValueError(msg)

    return ppf


def qdf_from_l_moments(
    lmbda: onp.ToFloat1D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    validate: bool = True,
    extrapolate: bool = False,
) -> _Fn1:
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
        msg = f"at least 2 L-moments required, got len(lmbda) = {_n}"
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

    def qdf(u: _T_x, *, r_max: int = -1) -> _T_x:
        """Quantile Distribution Function (QDF) polynomial approximation."""
        y = np.asanyarray(u, dtype=np.float64)
        # TODO: make this lazy
        y = np.where((y < 0) | (y > 1), np.nan, 2 * y - 1)

        c_ = c[:r_max] if 0 < r_max < len(c) else c

        x = fourier_jacobi(y, c_, alpha, beta)
        if extrapolate and _n > 2:
            x = (x + fourier_jacobi(y, c_[:-1], alpha, beta)) / 2

        return x[()]  # pyright: ignore[reportReturnType]

    if validate and np.any(qdf(plotting_positions(100)) < 0):
        msg = "QDF is not positive; consider increasing the trim"
        raise ValueError(msg)

    return qdf
