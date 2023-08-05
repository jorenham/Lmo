# pyright: reportIncompatibleMethodOverride=false

__all__ = ('l_rv',)

from collections.abc import Mapping
from typing import Any, Final, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.stats.distributions import rv_continuous  # type: ignore

from . import _poly as pu
from ._utils import clean_trim
from .diagnostic import l_ratio_bounds
from .typing import AnyTrim, PolySeries

T = TypeVar('T', bound=np.floating[Any])


def _ppf_poly_series(
    l_r: npt.NDArray[np.floating[Any]],
    s: float,
    t: float,
) -> PolySeries:
    r0 = np.arange(len(l_r), dtype=np.int_)
    c = l_r * r0 * (2 * r0 + s + t + 1) / (r0 + s + t)

    return pu.jacobi_series(c, t, s, domain=[0, 1], symbol='q')


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

    a: Final[float]
    b: Final[float]

    _trim: Final[tuple[int, int] | tuple[float, float]]

    def __init__(
        self,
        l_moments: npt.ArrayLike,
        trim: AnyTrim = (0, 0),
        *,
        deg_cdf: int | None = None,
        q_size: int = 200,
        q_alpha: float = .3,
        q_beta: float = .3,
        **kwargs: Any,
    ) -> None:
        r"""
        Parameters:
            l_moments: Array-like with the first L-moments, with order
                $r = 1, 2, ..., R$.
            trim: The left ($s$) and right ($t$) trim lengths of `l_moments`.

        Other Parameters:
            deg_cdf: The degree of the CDF polynomial. Defaults to $R-1$
            q_size: The amount of quantiles per degree to fit the CDF with.
                Defaults to 200.
            q_alpha: Percentile-point parameter for fitting the CDF.
                Defaults to 0.3
            q_beta: Percentile-point parameter for fitting the CDF.
                Defaults to 0.3
        """
        self._init_params = {
            'l_moments': l_moments,
            'trim': trim,
            'deg_cdf': deg_cdf,
            'q_size': q_size,
            'q_alpha': q_alpha,
            'q_beta': q_beta,
        }

        self._lm0 = l_r = np.asarray_chkfinite(l_moments, np.float_)
        l_r.setflags(write=False)

        if l_r.ndim != 1 and (l_r := l_r.squeeze()).ndim != 1:
            msg = f'l_moments must be 1-D, but its shape is {l_r.shape}'
            raise ValueError(msg)
        if l_r[1] <= 0:
            msg = f'the l-scale (at index 1) must be positive, got {l_r[1]}'
            raise ValueError(msg)

        self._trim = (s, t) = clean_trim(trim)

        self._lm = l_r = np.trim_zeros(l_r * (np.abs(l_r) > 1e-13), 'b')
        n = len(l_r)

        _check_lmoments(l_r, s, t)

        self._ppf_poly = _ppf = _ppf_poly_series(l_r, s, t)

        # empirical percentile points / plotting positions
        q = (
            (np.arange(q_size * n) + 1 - q_alpha)
            / (q_size * n + 1 - q_alpha - q_beta)
        )
        # mean-square-convergence sample weights (cheat a bit by using the
        # unstretched, to prevent streched values from getting 0 weight)
        w = self._weights(q)
        # generate samples
        x = self._ppf_poly(q)

        # weighted least-squares fit, using the same basis as the ppf
        deg = deg_cdf or _ppf.degree()
        self._cdf_poly = _cdf = _ppf.fit(x, q, deg, None, w=w)

        # pdf(x) = cdf'(x)
        self._pdf_poly = _pdf = _cdf.deriv(1)

        # figure out the support; the roots at 0 and 1, that are closest to
        # the left and right from the median, respectively
        med = _ppf(.5)
        a = np.r_[-np.inf, (_x0 := pu.roots(_cdf))[_x0 < med]][-1]
        b = np.r_[(_x1 := pu.roots(_cdf) - 1)[_x1 > med], np.inf][0]
        self.a, self.b = a, b

        # find the max of the pdf (the mode/modal values)
        _xs_max = (_m := pu.maxima(_pdf))[(_m > a) & (_m < b)]
        _fs_max = _pdf(_xs_max)
        self._mode = _xs_max[_fs_max == np.max(_fs_max)]
        self._fmax = np.max(_fs_max)

        # survival function: 1 - cdf(x)
        self._sf_poly = 1 - _ppf
        # inverse survival function: isf(q) = ppf(1 - q)
        self._isf_poly = _ppf(1 - _ppf.identity(domain=[0, 1]))

        super().__init__(  # type: ignore [reportUnknownMemberType]
            momtype=1,
            a=a,
            b=b,
            **kwargs,
        )

    @property
    def l_moments(self) -> npt.NDArray[np.float_]:
        r"""Initial L-moments, for orders $r = 1, 2, \dots, R$."""
        return self._lm0

    @property
    def trim(self) -> tuple[int, int] | tuple[float, float]:
        """The provided trim-lengths $(s, t)$."""
        return self._trim

    @property
    def pdf_poly(self) -> PolySeries:
        """Probability Density Function (PDF) polynomial."""
        return self._pdf_poly

    @property
    def cdf_poly(self) -> PolySeries:
        """Cumulative Density Function (CDF) polynomial."""
        return self._cdf_poly

    @property
    def ppf_poly(self) -> PolySeries:
        """Percent point function (PPF) polynomial."""
        return self._ppf_poly

    def _weights(self, q: npt.ArrayLike) -> npt.NDArray[np.float_]:
        _q = np.asarray(q, np.float_)
        s, t = self._trim
        return np.where(
            (_q >= 0) & (_q <= 1),
            _q ** s * (1 - _q) ** t,
            cast(float, self.badvalue),  # type: ignore
        )

    def _pdf(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float_]:
        return np.where(
            (x > self.a) & (x < self.b),
            np.clip(self._pdf_poly(x), 0, self._fmax),
            0,
        )[()]

    def _cdf(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float_]:
        return np.where(
            (x > self.a) & (x < self.b),
            np.clip(self._cdf_poly(x), 0, 1),
            0,
        )[()]

    def _sf(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float_]:
        return np.where(
            (x > self.a) & (x < self.b),
            np.clip(self._sf_poly(x), 0, 1),
            0,
        )[()]

    def _ppf(self, q: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return cast(npt.NDArray[np.float_], self._ppf_poly(q))

    def _isf(self, q: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return cast(npt.NDArray[np.float_], self._isf_poly(q))

    def _munp(self, n: int):
        return (self._ppf_poly**n).integ(lbnd=0)(1)

    def _updated_ctor_param(self) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            super()._updated_ctor_param() | self._init_params,
        )
