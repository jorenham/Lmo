# pyright: reportIncompatibleMethodOverride=false

__all__ = ('l_rv',)

import functools
import warnings
from collections.abc import Mapping
from typing import Any, Final, TypeVar, cast

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt
from scipy.stats.distributions import rv_continuous  # type: ignore

from . import _poly as pu
from ._utils import clean_trim
from .diagnostic import l_ratio_bounds
from .typing import AnyTrim, FloatVector, PolySeries

T = TypeVar('T', bound=np.floating[Any])


def _check_lmoments(l_r: npt.NDArray[np.floating[Any]], s: float, t: float):
    if (n := len(l_r)) < 2:
        msg = f'at least 2 L-moments required, got {n}'
        raise ValueError(msg)

    r = np.arange(1, n + 1, dtype=np.int_)
    t_r = l_r[2:] / l_r[1]
    t_r_max = l_ratio_bounds(r[2:], (s, t))
    if np.any(rs0_oob := np.abs(t_r) > t_r_max):
        r_oob = np.argwhere(rs0_oob)[0] + 3
        t_oob = t_r[rs0_oob][0]
        t_max = t_r_max[rs0_oob][0]
        msg = (
            f'invalid L-moment ratio for r={list(r_oob)}: '
            f'|{t_oob}| <= {t_max} does not hold'
        )
        raise ArithmeticError(msg)


def _ppf_poly_series(
    l_r: npt.NDArray[np.floating[Any]],
    s: float,
    t: float,
) -> PolySeries:
    r0 = np.arange(len(l_r), dtype=np.int_)

    c = 2 * r0 + 1
    if s or t:
        c = (c + s + t) * r0 / (r0 + s + t)

    return pu.jacobi_series(
        c * l_r,
        t,
        s,
        domain=[0, 1],
        # convert to Legendre, even if trimmed; this avoids huge coeficient
        kind=npp.Legendre,
        symbol='q',
    )


class l_rv(rv_continuous):  # noqa: N801
    r"""
    Estimate a distribution using the given L-moments.
    See [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] for the
    available method.

    The PPF (quantile function) is estimated using an exactly defined
    (Jacobi) polynomial series. The CDF is also a polynomial, but is estimated
    using inverse regression on the PPF.

    Hosking showed in 2007 that

    $$
    \hat{Q}(q) = \sum_{r=1}^{R}
        \frac{(r - 1) (2r + s + t - 1)}{r + s + t - 1}
        \lambda^{(s, t)}_r
        P^{(t, s)}_{r - 1}(2u - 1)
    $$

    converges to the quantile function $Q(q)$ in the weighted mean-squared
    sense:

    $$
    \lim_{R \rightarrow \infty}
    \int_0^1 u^s (1-u)^t
    \Big( \hat{Q}(q) - Q(u) \Big)^2 du
    = 0
    $$
    """

    _lm: Final[npt.NDArray[np.floating[Any]]]
    _trim: Final[tuple[int, int] | tuple[float, float]]

    _ppf_poly: Final[PolySeries]
    _isf_poly: Final[PolySeries]

    a: float
    b: float
    badvalue: float = np.nan

    def __init__(
        self,
        l_moments: FloatVector,
        trim: AnyTrim = (0, 0),
        a: float = -np.inf,
        b: float = np.inf,
    ) -> None:
        r"""
        Args:
            l_moments:
                Vector containing the first $R$ consecutive L-moments
                $\left[
                \lambda^{(s, t)}_1 \;
                \lambda^{(s, t)}_2 \;
                \dots \;
                \lambda^{(s, t)}_R
                \right]$, where $R \ge 2$.

                Sample L-moments can be estimated using e.g.
                `lmo.l_moment(x, np.mgrid[:R] + 1, trim=(s, t))`.

                The trim-lengths $(s, t)$ should be the same for all
                L-moments.
            trim:
                The left and right trim-lengths $(s, t)$, that correspond
                to the provided `l_moments`.
            a:
                Lower bound of the support of the distribution.
                By default it is estimated from the L-moments.
            b:
                Upper bound of the support of the distribution.
                By default it is estimated from the L-moments.

        Raises:
            ValueError: If `len(l_moments) < 2`, `l_moments.ndim != 1`, or
                there are invalid L-moments / trim-lengths.

        """
        l_r = np.asarray_chkfinite(l_moments)
        l_r.setflags(write=False)

        self._trim = (s, t) = clean_trim(trim)

        _check_lmoments(l_r, s, t)
        self._lm = l_r


        # quantile function (inverse of cdf)
        ppf = _ppf_poly_series(l_r, s, t)
        self._tol = np.finfo(ppf.coef.dtype).eps
        self._ppf_poly = ppf = ppf.trim(self._tol)

        # inverse survival function
        self._isf_poly = ppf(1 - ppf.identity(domain=[0, 1])).trim(self._tol)

        # empirical support
        q0, q1 = ppf(np.array([0, 1]))
        assert q0 < q1, (q0, q1)

        super().__init__(  # type: ignore [reportUnknownMemberType]
            momtype=1,
            a=max(a, q0),
            b=min(b, q1),
            name='l_nonparametric',
        )

    @property
    def l_moments(self) -> npt.NDArray[np.float_]:
        r"""Initial L-moments, for orders $r = 1, 2, \dots, R$."""
        return self._lm

    @property
    def trim(self) -> tuple[int, int] | tuple[float, float]:
        """The provided trim-lengths $(s, t)$."""
        return self._trim

    @property
    def ppf_poly(self) -> PolySeries:
        r"""
        Polynomial estimate of the percent point function (PPF), a.k.a.
        the quantile function (QF), or the inverse cumulative distribution
        function (ICDF).

        Note:
            Converges to the "true" PPF in the mean-squared sense, with
            weight function $q^s (1 - q)^t$ of quantile $q \in \[0, 1\]$,
            and trim-lengths $(t_1, t_2) \in \mathbb{R^+} \times \mathbb{R^+}$.

        Returns:
            A [`numpy.polynomial.Legendre`][numpy.polynomial.legendre.Legendre]
                orthogonal polynomial series instance.
        """
        return self._ppf_poly

    @functools.cached_property
    def cdf_poly(self) -> PolySeries:
        """
        Polynomial least-squares interpolation of the CDF.

        Returns:
            A [`numpy.polynomial.Legendre`][numpy.polynomial.legendre.Legendre]
                orthogonal polynomial series instance.
        """
        ppf = self._ppf_poly
        # number of variables of the PPF poly
        k0 = ppf.degree() + 1
        assert k0 > 1

        n = max(100, k0 * 10)
        x = np.linspace(self.a, self.b, n)
        q = cast(npt.NDArray[np.float_], self.cdf(x))  # type: ignore
        y = ppf.deriv()(q)
        w = np.sqrt(self._weights(q) + 0.01)

        # choose the polynomial that minimizes the BIC
        bic_min = np.inf
        cdf_best = None
        for k in range(max(k0 // 2, 2), k0 + max(k0 // 2, 8)):
            # fit
            cdf = ppf.fit(x, q, k - 1).trim(self._tol)
            k = cdf.degree() + 1

            # according to the inverse function theorem, this should be 0
            eps = 1 / cdf.deriv()(x) - y

            # Bayesian information criterion (BIC)
            bic = (
                (k - 1) * np.log(n)
                + n * np.log(np.average(eps**2, weights=w))
            )

            # minimize the BIC
            if bic < bic_min:
                bic_min = bic
                cdf_best = cdf

        assert cdf_best is not None
        return cdf_best

    @functools.cached_property
    def pdf_poly(self) -> PolySeries:
        """
        Derivative of the polynomial interpolation of the CDF, i.e. the
        polynomial estimate of the PDF.

        Returns:
            A [`numpy.polynomial.Legendre`][numpy.polynomial.legendre.Legendre]
                orthogonal polynomial series instance.
        """
        return self.cdf_poly.deriv()

    def _weights(self, q: npt.ArrayLike) -> npt.NDArray[np.float_]:
        _q = np.asarray(q, np.float_)
        s, t = self._trim
        return np.where(
            (_q >= 0) & (_q <= 1),
            _q ** s * (1 - _q) ** t,
            cast(float, getattr(self, 'badvalue', np.nan)),  # type: ignore
        )

    def _ppf(self, q: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return cast(npt.NDArray[np.float_], self._ppf_poly(q))

    def _isf(self, q: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return cast(npt.NDArray[np.float_], self._isf_poly(q))

    def _cdf_single(self, x: float) -> float:
        # find all q where Q(q) == x
        q0 = pu.roots(self._ppf_poly - x)

        if (n := len(q0)) == 0:
            return self.badvalue
        if n > 1:
            warnings.warn(
                f'multiple fixed points at {x = :.6f}: '
                f'{list(np.round(q0, 6))}',
                stacklevel=3,
            )

            if np.ptp(q0) <= 1/4:
                # "close enough" if within the same quartile;
                # probability-weighted interpolation
                return np.average(q0, weights=q0 * (1 - q0))  # type: ignore

            return self.badvalue

        return q0[0]

    def _pdf(self, x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return np.clip(cast(npt.NDArray[np.float_], self.pdf_poly(x)), 0, 1)

    def _munp(self, n: int):
        # non-central product-moment $E[X^n]$
        return (self._ppf_poly**n).integ(lbnd=0)(1)

    def _updated_ctor_param(self) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            super()._updated_ctor_param() | {
                'l_moments': self._lm,
                'trim': self._trim,
            },
        )
