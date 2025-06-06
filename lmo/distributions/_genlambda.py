"""Probability distributions, compatible with [`scipy.stats`][scipy.stats]."""

from __future__ import annotations

import functools
import math
import sys
from typing import Final, TypeAlias, TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc
import scipy.special as sps
from scipy.stats.distributions import rv_continuous

import lmo.typing as lmt
from lmo.special import harmonic
from lmo.theoretical import entropy_from_qdf, l_moment_from_ppf
from ._lm import lm_genlambda

if sys.version_info >= (3, 13):
    from typing import override
else:
    from typing_extensions import override


__all__ = ("genlambda_gen",)


_F8: TypeAlias = float | np.float64
_ArrF8: TypeAlias = onp.ArrayND[np.float64]

_XT = TypeVar("_XT", bound=_F8 | _ArrF8)


def _genlambda_support(b: float, d: float, f: float) -> tuple[float, float]:
    xa = -(1 + f) / b if b > 0 else -math.inf
    xb = (1 - f) / d if d > 0 else math.inf
    return xa, xb


def _genlambda_ppf0(q: float, b: float, d: float, f: float) -> float:
    """PPF of the GLD."""
    if math.isnan(q):
        return math.nan
    if q <= 0:
        return _genlambda_support(b, d, f)[0]
    if q >= 1:
        return _genlambda_support(b, d, f)[1]

    u = math.log(q) if b == 0 else (q**b - 1) / b
    v = math.log(1 - q) if d == 0 else ((1 - q) ** d - 1) / d
    return (1 + f) * u - (1 - f) * v


_genlambda_ppf: Final = np.vectorize(_genlambda_ppf0, [float])


@np.errstate(divide="ignore")
def _genlambda_qdf(q: _XT, b: float, d: float, f: float) -> _XT:
    return (1 + f) * q ** (b - 1) + (1 - f) * (1 - q) ** (d - 1)  # pyright: ignore[reportReturnType]


# pyright: reportIncompatibleMethodOverride=false


def _genlambda_cdf0(  # noqa: C901
    x: float,
    b: float,
    d: float,
    f: float,
    *,
    ptol: float = 1e-04,
    xtol: float = 1e-14,
    maxiter: int = 60,
) -> float:
    """
    Compute the CDF of the GLD using bracketing search with special checks.

    Uses the same (unnamed?) algorithm as `scipy.special.tklmbda`:
    https://github.com/scipy/scipy/blob/v1.11.4/scipy/special/cephes/tukey.c
    """
    if math.isnan(x) or math.isnan(b) or math.isnan(d) or math.isnan(f):
        return math.nan

    # extrema
    xa, xb = _genlambda_support(b, d, f)
    if x <= xa:
        return 0
    if x >= xb:
        return 1

    # special cases
    if abs(f + 1) < ptol:
        if d == 0:
            return 1 - math.exp(-x / 2)

        return 1 - (1 - d * x / 2) ** (1 / d)
    if abs(f - 1) < ptol:
        return math.exp(x / 2) if b == 0 else (1 + b * x / 2) ** (1 / b)
    if abs(f) < ptol and abs(b) < ptol and abs(d) < ptol:
        # logistic
        if x >= 0:
            return 1 / (1 + math.exp(-x))

        return math.exp(x) / (1 + math.exp(x))
    if abs(b - 1) < ptol and abs(d - 1) < ptol:
        # uniform on [-1 - f, 1 - f]
        return (x + f + 1) / 2

    # bracketing search, using a similar algorithm as `scipy.special.tklmbda`
    p_low, p_mid, p_high = 0.0, 0.5, 1.0
    for _ in range(maxiter):
        x_eval = _genlambda_ppf0(p_mid, b, d, f)
        if abs(x_eval - x) <= xtol:
            break

        if x_eval > x:
            p_mid, p_high = (p_mid + p_low) / 2, p_mid
        else:
            p_mid, p_low = (p_mid + p_high) / 2, p_mid

        if abs(p_mid - p_low) <= xtol:
            break

    return p_mid


_genlambda_cdf: Final = np.vectorize(
    _genlambda_cdf0,
    [float],
    excluded={"ptol", "xtol", "maxiter"},
)


class genlambda_gen(rv_continuous):
    @override
    def _argcheck(self, /, b: _F8, d: _F8, f: _F8) -> np.bool_:
        return np.isfinite(b) & np.isfinite(d) & (f >= -1) & (f <= 1)

    @override
    def _shape_info(self, /) -> list[lmt.ShapeInfo]:
        ibeta = lmt.ShapeInfo("b", False, (-np.inf, np.inf), (False, False))
        idelta = lmt.ShapeInfo("d", False, (-np.inf, np.inf), (False, False))
        iphi = lmt.ShapeInfo("f", False, (-1, 1), (True, True))
        return [ibeta, idelta, iphi]

    @override
    def _get_support(self, /, b: float, d: float, f: float) -> tuple[float, float]:
        return _genlambda_support(b, d, f)

    @override
    def _fitstart(
        self,
        /,
        data: _ArrF8,
        args: tuple[float, float, float] | None = None,
    ) -> tuple[_F8, _F8, _F8, _F8, _F8]:
        #  Arbitrary, but the default f=1 is a bad start
        loc, scale, b, d, f = super()._fitstart(data, args or (1.0, 1.0, 0.0))
        return loc, scale, b, d, f

    @override
    def _pdf(self, /, x: _XT, b: float, d: float, f: float) -> _XT:
        return 1 / self._qdf(self._cdf(x, b, d, f), b, d, f)  # pyright: ignore[reportReturnType]

    @override
    def _cdf(self, /, x: _XT, b: float, d: float, f: float) -> _XT:
        return _genlambda_cdf(x, b, d, f)

    def _qdf(self, /, q: _XT, b: float, d: float, f: float) -> _XT:
        return _genlambda_qdf(q, b, d, f)

    @override
    def _ppf(self, /, q: _XT, b: float, d: float, f: float) -> _XT:
        return _genlambda_ppf(q, b, d, f)

    @override
    def _stats(
        self,
        b: float,
        d: float,
        f: float,
    ) -> tuple[float, float, float | None, float | None]:
        if b <= -1 or d <= -1:
            # hard NaN (not inf); indeterminate integral
            return math.nan, math.nan, math.nan, math.nan

        a, c = 1 + f, 1 - f
        b1, d1 = 1 + b, 1 + d

        m1: float = 0.0 if b == d and f == 0 else lm_genlambda(1, 0, 0, b, d, f).item()

        if b <= -1 / 2 or d <= -1 / 2:
            return m1, math.nan, math.nan, math.nan

        if b == d == 0:
            m2 = 4 * f**2 + math.pi**2 * (1 - f**2) / 3
        elif b == 0:
            m2 = (
                a**2
                + (c / d1) ** 2 / (d1 + d)
                + 2 * a * c / (d * d1) * (1 - harmonic(1 + d))
            )
        elif d == 0:
            m2 = (
                c**2
                + (a / b1) ** 2 / (b1 + b)
                + 2 * a * c / (b * b1) * (1 - harmonic(1 + b))
            )
        else:
            m2 = (
                (a / b1) ** 2 / (b1 + b)
                + (c / d1) ** 2 / (d1 + d)
                + 2 * a * c / (b * d) * (1 / (b1 * d1) - sps.beta(b1, d1).item())
            )

        # Feeling adventurous? You're welcome to contribute these missing
        # skewness and kurtosis stats here :)
        if b <= -1 / 3 or d <= -1 / 3:
            return m1, m2, math.nan, math.nan
        m3 = None

        if b <= -1 / 4 or d <= -1 / 4:
            return m1, m2, m3, math.nan
        m4 = None

        return m1, m2, m3, m4

    def _entropy(self, b: float, d: float, f: float) -> float:
        return entropy_from_qdf(_genlambda_qdf, b, d, f)

    def _l_moment(
        self,
        r: onp.ArrayND[npc.integer],
        b: float,
        d: float,
        f: float,
        *,
        trim: tuple[int, int] | tuple[float, float],
        quad_opts: lmt.QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim

        if quad_opts is not None:
            # only do numerical integration when quad_opts is passed
            lmbda_r = l_moment_from_ppf(
                functools.partial(self._ppf, b=b, d=d, f=f),
                r,
                trim=trim,
                quad_opts=quad_opts,
            )
            return np.asarray(lmbda_r)

        return lm_genlambda(r, s, t, b, d, f)
