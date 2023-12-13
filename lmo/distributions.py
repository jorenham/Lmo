"""Probability distributions, compatible with [`scipy.stats`][scipy.stats]."""
__all__ = (
    'l_rv_nonparametric',
    'kumaraswamy',
    'wakeby',
    'genlambda',
)

# pyright: reportIncompatibleMethodOverride=false

import functools
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    Final,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    cast,
)

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt
import scipy.special as sc  # type: ignore
from scipy.stats._distn_infrastructure import _ShapeInfo  # type: ignore
from scipy.stats.distributions import (  # type: ignore
    rv_continuous as _rv_continuous,
)

from ._poly import jacobi_series, roots
from ._utils import (
    clean_order,
    clean_trim,
)
from .diagnostic import l_ratio_bounds
from .special import harmonic
from .theoretical import entropy_from_qdf, l_moment_from_ppf
from .typing import (
    AnyTrim,
    FloatVector,
    PolySeries,
    QuadOptions,
    RVContinuous,
)

T = TypeVar('T')
X = TypeVar('X', bound='l_rv_nonparametric')
F = TypeVar('F', bound=np.floating[Any])
M = TypeVar('M', bound=Callable[..., Any])
V = TypeVar('V', bound=float | npt.NDArray[np.float64])


_F_EPS: Final[np.float64] = np.finfo(float).eps


# Non-parametric

def _check_lmoments(l_r: npt.NDArray[np.floating[Any]], s: float, t: float):
    if (n := len(l_r)) < 2:
        msg = f'at least 2 L-moments required, got {n}'
        raise ValueError(msg)
    if n == 2:
        return

    r = np.arange(1, n + 1)
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
    # Corrected version of Theorem 3. from Hosking (2007).
    #
    r = np.arange(1, len(l_r) + 1)
    c = (s + t - 1 + 2 * r) * r / (s + t + r)

    return jacobi_series(
        c * l_r,
        t,
        s,
        domain=[0, 1],
        # convert to Legendre, even if trimmed; this avoids huge coefficient
        kind=npp.Legendre,
        symbol='q',
    )

class l_rv_nonparametric(_rv_continuous):  # noqa: N801
    r"""
    Estimate a distribution using the given L-moments.
    See [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] for the
    available method.

    The PPF (quantile function) is estimated using generalized Fourier series,
    with the (shifted) Jacobi orthogonal polynomials as basis, and the (scaled)
    L-moments as coefficients.

    The *corrected* version of theorem 3 from Hosking (2007) states that

    $$
    \hat{Q}(q) = \sum_{r=1}^{R}
        \frac{(r + 1) (2r + s + t - 1)}{r + s + t + 1}
        \lambda^{(s, t)}_r
        P^{(t, s)}_{r - 1}(2u - 1) \; ,
    $$

    converges almost everywhere as $R \rightarrow \infty$, for any
    sufficiently smooth (quantile) function $Q(u)$ with $0 < u < 1$.

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
        - [Wolfram Research - Jacobi polynomial Fourier Expansion](
            http://functions.wolfram.com/05.06.25.0007.01)

    See Also:
        - [Generalized Fourier series - Wikipedia](
            https://wikipedia.org/wiki/Generalized_Fourier_series)
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
        a: float | None = None,
        b: float | None = None,
        **kwargs: Any,
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
            **kwargs:
                Optional params for `scipy.stats.rv_continuous`.

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
        self._ppf_poly = ppf = _ppf_poly_series(l_r, s, t).trim(_F_EPS)

        # inverse survival function
        self._isf_poly = ppf(1 - ppf.identity(domain=[0, 1])).trim(_F_EPS)

        # empirical support
        self._a0, self._b0 = (q0, q1) = ppf(np.array([0, 1]))
        if q0 >= q1:
            msg = 'invalid l_rv_nonparametric: ppf(0) >= ppf(1)'
            raise ArithmeticError(msg)

        kwargs.setdefault('momtype', 1)
        super().__init__(  # type: ignore [reportUnknownMemberType]
            a=q0 if a is None else a,
            b=q1 if b is None else b,
            **kwargs,
        )

    @property
    def l_moments(self) -> npt.NDArray[np.float64]:
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
        q = cast(npt.NDArray[np.float64], self.cdf(x))  # type: ignore
        y = ppf.deriv()(q)
        w = np.sqrt(self._weights(q) + 0.01)

        # choose the polynomial that minimizes the BIC
        bic_min = np.inf
        cdf_best = None
        for k in range(max(k0 // 2, 2), k0 + max(k0 // 2, 8)):
            # fit
            cdf = ppf.fit(x, q, k - 1).trim(_F_EPS)
            k = cdf.degree() + 1

            # according to the inverse function theorem, this should be 0
            eps = 1 / cdf.deriv()(x) - y

            # Bayesian information criterion (BIC)
            bic = (k - 1) * np.log(n) + n * np.log(
                np.average(eps**2, weights=w),
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

    def _weights(self, q: npt.ArrayLike) -> npt.NDArray[np.float64]:
        _q = np.asarray(q, np.float64)
        s, t = self._trim
        return np.where(
            (_q >= 0) & (_q <= 1),
            _q**s * (1 - _q) ** t,
            cast(float, getattr(self, 'badvalue', np.nan)),  # type: ignore
        )

    def _ppf(self, q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self._ppf_poly(q))

    def _isf(self, q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self._isf_poly(q))

    def _cdf_single(self, x: float) -> float:
        # find all q where Q(q) == x
        q0 = roots(self._ppf_poly - x)

        if (n := len(q0)) == 0:
            return self.badvalue
        if n > 1:
            warnings.warn(
                f'multiple fixed points at {x = :.6f}: '
                f'{list(np.round(q0, 6))}',
                stacklevel=3,
            )

            if cast(float, np.ptp(q0)) <= 1 / 4:
                # "close enough" if within the same quartile;
                # probability-weighted interpolation
                return np.average(q0, weights=q0 * (1 - q0))  # type: ignore

            return self.badvalue

        return q0[0]

    def _pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.clip(cast(npt.NDArray[np.float64], self.pdf_poly(x)), 0, 1)

    def _munp(self, n: int):
        # non-central product-moment $E[X^n]$
        return (self._ppf_poly**n).integ(lbnd=0)(1)

    def _updated_ctor_param(self) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            super()._updated_ctor_param()
            | {
                'l_moments': self._lm,
                'trim': self._trim,
            },
        )

    @classmethod
    def fit(
        cls,
        data: npt.ArrayLike,
        /,
        rmax: SupportsIndex | None = None,
        trim: AnyTrim = (0, 0),
    ) -> 'l_rv_nonparametric':
        r"""
        Estimate L-moment from the samples, and return a new
        `l_rv_nonparametric` instance.

        Args:
            data:
                1d array-like with univariate sample observations.
            rmax:
                The (maximum) amount of L-moment orders to use.
                Defaults to $\lceil 4 \log_{10} N \rceil$.
                The quantile polynomial will be of degree `rmax - 1`.
            trim:
                The left and right trim-lengths $(s, t)$, that correspond
                to the provided `l_moments`.

        Returns:
            A fitted
            [`l_rv_nonparametric`][lmo.distributions.l_rv_nonparametric]
            instance.

        Todo:
            - Optimal `rmax` selection (the error appears to be periodic..?)
            - Optimal `trim` selection
        """
        # avoid circular imports
        from ._lm import l_moment

        # x needs to be sorted anyway
        x: npt.NDArray[np.floating[Any]] = np.sort(data)

        a, b = x[[0, -1]]

        if rmax is None:
            _rmax = math.ceil(np.log10(x.size) * 4)
        else:
            _rmax = clean_order(rmax, name='rmax', rmin=2)

        _trim = clean_trim(trim)

        # sort kind 'stable' if already sorted
        l_r = l_moment(
            x,
            np.arange(1, _rmax + 1),
            trim=_trim,
            sort='stable',  # stable sort if fastest if already sorted
        )

        return cls(l_r, trim=_trim, a=a, b=b)


# Parametric


_ArrF8: TypeAlias = npt.NDArray[np.float64]

def _kumaraswamy_lmo0(
    r: int,
    s: int,
    t: int,
    a: float,
    b: float,
) -> float:
    if r == 0:
        return 1.0

    k = np.arange(t + 1, r + s + t + 1)
    return (
        (-1)**(k - 1)
        * cast(_ArrF8, sc.comb(r + k - 2, r + t - 1))  # type: ignore
        * cast(_ArrF8, sc.comb(r + s + t, k))  # type: ignore
        * cast(_ArrF8, sc.beta(1 / a, 1 + k * b)) / a  # type: ignore
    ).sum() / r

_kumaraswamy_lmo = np.vectorize(_kumaraswamy_lmo0, [float], excluded={1, 2})


class kumaraswamy_gen(_rv_continuous):  # noqa: N801
    def _argcheck(self, a: float, b: float) -> bool:
        return (a > 0) & (b > 0)

    def _shape_info(self) -> Sequence[_ShapeInfo]:
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ia, ib]

    def _pdf(
        self,
        x: npt.NDArray[np.float64],
        a: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        return a * b * x**(a - 1) * (1 - x**a)**(b - 1)

    def _logpdf(
        self,
        x: npt.NDArray[np.float64],
        a: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        return (
            np.log(a * b)
            + (a - 1) * np.log(x)
            + (b - 1) * np.log(1 - x**a)
        )

    def _cdf(
        self,
        x: npt.NDArray[np.float64],
        a: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        return 1 - (1 - x**a)**b

    def _sf(
        self,
        x: npt.NDArray[np.float64],
        a: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        return (1 - x**a)**(b - 1)

    def _isf(
        self,
        q: npt.NDArray[np.float64],
        a: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        return (1 - q**(1 / b))**(1 / a)

    def _ppf(
        self,
        q: npt.NDArray[np.float64],
        a: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        return (1 - (1 - q)**(1 / b))**(1 / a)

    def _entropy(self, a: float, b: float) -> float:
        # https://en.wikipedia.org/wiki/Kumaraswamy_distribution
        return (1 - 1 / b) + (1 - 1 / a) * harmonic(b) - np.log(a * b)

    def _munp(
        self,
        n: int,
        a: float,
        b: float,
    ) -> float:
        return b * cast(float, sc.beta(1 + n / a, b))  # type: ignore

    def _l_moment(
        self,
        r: npt.NDArray[np.int64],
        a: float,
        b: float,
        trim: tuple[int, int] | tuple[float, float],
        quad_opts: QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim
        if quad_opts is not None or isinstance(s, float):
            return cast(
                _ArrF8,
                super()._l_moment(  # type: ignore
                    r,
                    a,
                    b,
                    trim=trim,
                    quad_opts=quad_opts,
                ),
            )

        return np.atleast_1d(cast(_ArrF8, _kumaraswamy_lmo(r, s, t, a, b)))

kumaraswamy: RVContinuous[float, float] = kumaraswamy_gen(
    a=0.0,
    b=1.0,
    name='kumaraswamy',
)  # type: ignore
r"""
A Kumaraswamy random variable, similar to
[`scipy.stats.beta`][scipy.stats.beta].

The probability density function for
[`kumaraswamy`][lmo.distributions.kumaraswamy] is:

\[
    f(x, a, b) = a x^{a - 1} b \left(1 - x^a\right)^{b - 1}
\]

for \( 0 < x < 1,\ a > 0,\ b > 0 \).

[`kumaraswamy`][kumaraswamy] takes \( a \) and \( b \) as shape parameters.

See Also:
    - [Theoretical L-moments - Kumaraswamy](distributions.md#kumaraswamy)

"""


def _wakeby_ub(b: float, d: float, f: float) -> float:
    """Upper bound of x."""
    if d < 0:
        return f / b - (1 - f) / d
    if f == 1 and b:
        return 1 / b
    return math.inf


def _wakeby_isf0(
    q: float,
    b: float,
    d: float,
    f: float,
) -> float:
    """Inverse survival function, does not validate params."""
    if q <= 0:
        return _wakeby_ub(b, d, f)
    if q >= 1:
        return 0.

    if f == 0:
        u = 0.
    elif b == 0:
        u = math.log(q)
    else:
        u = (q**b - 1) / b

    if f == 1:
        v = 0.
    elif d == 0:
        v = u if b == 0 and f != 0 else math.log(q)
    else:
        v = -(q**(-d) - 1) / d

    return -f * u - (1 - f) * v

_wakeby_isf = np.vectorize(_wakeby_isf0, [float])


def _wakeby_qdf(
    p: npt.NDArray[np.float64],
    b: float,
    d: float,
    f: float,
) -> npt.NDArray[np.float64]:
    """Quantile density function (QDF), the derivative of the PPF."""
    q = 1 - p
    return f * q**(b - 1) + (1 - f) * q**(-d - 1)


def _wakeby_sf0(  # noqa: C901
    x: float,
    b: float,
    d: float,
    f: float,
) -> float:
    """
    Numerical approximation of Wakeby's survival function.

    Uses a modified version of Halley's algorithm, as originally implemented
    by J.R.M. Hosking in fortran: https://lib.stat.cmu.edu/general/lmoments
    """
    if x <= 0:
        return 1.

    if x >= _wakeby_ub(b, d, f):
        return 0.

    if b == f == 1:
        # standard uniform
        return 1 - x
    if b == d == 0:
        # standard exponential
        assert f == 1
        return math.exp(-x)
    if f == 1:
        # GPD (bounded above)
        return (1 - b * x)**(1 / b)
    if f == 0:
        # GPD (no upper bound)
        return (1 + d * x)**(-1 / d)
    if b == d and b > 0:
        # unnamed special case
        cx = b * x
        return (
            (2 * f - cx - 1 + math.sqrt((cx + 1)**2 - 4 * cx * f)) / (2 * f)
        )**(1 / b)
    if b == 0 and d != 0:
        # https://wikipedia.org/wiki/Lambert_W_function
        # it's easy to show that this is valid for all x, f, and d
        w = (1 - f) / f
        return (
            w / sc.lambertw(w * math.exp((1 + d * x) / f - 1))  # type: ignore
        )**(1 / d)

    if x < _wakeby_isf0(.9, b, d, f):
        z = 0
    elif x >= _wakeby_isf0(.01, b, d, f):
        if d < 0:
            z = math.log(1 + (x - f / b) * d / (1 - f)) / d
        elif d > 0:
            z = math.log(1 + x * d / (1 - f)) / d
        else:
            z = (x - f / b) / (1 - f)
    else:
        z = .7

    eps = 1e-8
    maxit = 50
    ufl = math.log(math.nextafter(0, 1))

    for _ in range(maxit):
        bz = -b * z
        eb = math.exp(bz) if bz >= ufl else 0
        gb = (1 - eb) / b if abs(b) > eps else z

        ed = math.exp(d * z)
        gd = (1 - ed) / d if abs(d) > eps else -z

        x_est = f * gb - (1 - f) * gd
        qd0 = x - x_est
        qd1 = f * eb + (1 - f) * ed
        qd2 = -f * b * eb + (1 - f) * d * ed

        tmp = qd1 - .5 * qd0 * qd2 / qd1
        if tmp <= 0:
            tmp = qd1

        z_inc = min(qd0 / tmp, 3)
        z_new = z + z_inc
        if z_new <= 0:
            z /= 5
            continue
        z = z_new

        if abs(z_inc) <= eps:
            break
    else:
        warnings.warn(
            'Wakeby SF did not converge, the result may be unreliable',
            RuntimeWarning,
            stacklevel=4,
        )

    return math.exp(-z) if -z >= ufl else 0


_wakeby_sf = np.vectorize(_wakeby_sf0, [float])

def _wakeby_lmo0(
    r: int,
    s: float,
    t: float,
    b: float,
    d: float,
    f: float,
) -> float:
    if r == 0:
        return 1

    if d >= (b == 0) + 1 + t:
        return math.nan

    def _lmo0_partial(theta: float, scale: float) -> float:
        if scale == 0:
            return 0
        if r == 1 and theta == 0:
            return cast(float, harmonic(s + t + 1) - harmonic(t))

        return scale * (
            sc.poch(r + t, s + 1)  # type: ignore
            * sc.poch(1 - theta, r - 2)  # type: ignore
            / sc.poch(1 + theta + t, r + s)  # type: ignore
            + (1 / theta if r == 1 else 0)
        ) / r

    return _lmo0_partial(b, f) + _lmo0_partial(-d, 1 - f)

_wakeby_lmo = np.vectorize(_wakeby_lmo0, [float], excluded={1, 2})

class wakeby_gen(_rv_continuous):  # noqa: N801
    a: float

    def _argcheck(self, b: float, d: float, f: float) -> int:
        return (
            np.isfinite(b)
            & np.isfinite(d)
            & (b + d >= 0)
            & ((b + d > 0) | (f == 1))
            & (f >= 0)
            & (f <= 1)
            & ((f > 0) | (b == 0))
            & ((f < 1) | (d == 0))
        )

    def _shape_info(self) -> Sequence[_ShapeInfo]:
        ibeta = _ShapeInfo('b', False, (-np.inf, np.inf), (False, False))
        idelta = _ShapeInfo('d', False, (-np.inf, np.inf), (False, False))
        iphi = _ShapeInfo('f', False, (0, 1), (True, True))
        return [ibeta, idelta, iphi]

    def _get_support(
        self,
        b: float,
        d: float,
        f: float,
    ) -> tuple[float, float]:
        if not self._argcheck(b, d, f):
            return math.nan, math.nan

        return self.a, _wakeby_ub(b, d, f)

    def _fitstart(
        self,
        data: npt.NDArray[np.float64],
        args: tuple[float, float, float] | None = None,
    ) -> tuple[float, float, float, float, float]:
        #  Arbitrary, but the default f=1 is a bad start
        return super()._fitstart(data, args or (1., 1., .5))  # type: ignore

    def _pdf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        # application of the inverse function theorem
        return 1 / _wakeby_qdf(self._cdf(x, b, d, f), b, d, f)

    def _cdf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return 1 - _wakeby_sf(x, b, d, f)

    def _ppf(
        self,
        p: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return _wakeby_isf(1 - p, b, d, f)

    def _isf(
        self,
        q: npt.NDArray[np.float64],
         b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return _wakeby_isf(q, b, d, f)

    def _stats(self, b: float, d: float, f: float) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        if d >= 1:
            # hard NaN (not inf); indeterminate integral
            return math.nan, math.nan, math.nan, math.nan

        u = f / (1 + b)
        v = (1 - f) / (1 - d)

        m1 = u + v

        if d >= 1 / 2:
            return m1, math.nan, math.nan, math.nan

        m2 = (
            u**2 / (1 + 2 * b)
            + 2 * u * v / (1 + b - d)
            + v**2 / (1 - 2 * d)
        )

        # no skewness and kurtosis (yet?); the equations are kinda huge...
        if d >= 1 / 3:
            return m1, m2, math.nan, math.nan
        m3 = None

        if d >= 1 / 4:
            return m1, m2, m3, math.nan
        m4 = None

        return m1, m2, m3, m4

    def _l_moment(
        self,
        r: npt.NDArray[np.int64],
        b: float,
        d: float,
        f: float,
        trim: tuple[int, int] | tuple[float, float],
        quad_opts: QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim

        if quad_opts is not None:
            # only do numerical integration when quad_opts is passed
            lmbda_r = cast(
                float | npt.NDArray[np.float64],
                l_moment_from_ppf(
                    functools.partial(self._ppf, b=b, d=d, f=f), # type: ignore
                    r,
                    trim=trim,
                    quad_opts=quad_opts,
                ),  # type: ignore
            )
            return np.asarray(lmbda_r)

        return np.atleast_1d(
            cast(_ArrF8, _wakeby_lmo(r, s, t, b, d, f)),
        )

    def _entropy(self, b: float, d: float, f: float) -> float:
        """
        Entropy can be calculated from the QDF (PPF derivative) as the
        Integrate[Log[QDF[u]], {u, 0, 1}]. This is the (semi) closed-form
        solution in this case.
        At the time of writing, this result appears to be novel.

        The `f` conditionals are the limiting cases, e.g. for uniform,
        exponential, and GPD (genpareto).
        """
        if f == 0:
            return 1 + d
        if f == 1:
            return 1 - b

        bd = b + d
        assert bd > 0

        return 1 - b + bd * cast(
            float,
            sc.hyp2f1(1, 1 / bd, 1 + 1 / bd, -f / (1 - f)),  # type: ignore
        )


wakeby: RVContinuous[float, float, float] = wakeby_gen(
    a=0.0,
    name='wakeby',
)  # type: ignore
r"""A Wakeby random variable, a generalization of
[`scipy.stats.genpareto`][scipy.stats.genpareto].

[`wakeby`][wakeby] takes \( b \), \( d \) and \( f \) as shape parameters.

For a detailed description of the Wakeby distribution, refer to
[Distributions - Wakeby](distributions.md#wakeby).
"""

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
    v = math.log(1 - q) if d == 0 else ((1 - q)**d - 1) / d
    return (1 + f) * u - (1 - f) * v

_genlambda_ppf = np.vectorize(_genlambda_ppf0, [float])

@np.errstate(divide='ignore')
def _genlambda_qdf(q: V, b: float, d: float, f: float) -> V:
    return cast(V, (1 + f) * q**(b - 1) + (1 - f) * (1 - q)**(d - 1))

def _genlambda_cdf0(  # noqa: C901
    x: float,
    b: float,
    d: float,
    f: float,
    *,
    ptol: float = 1e-4,
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
        return 1 - math.exp(-x / 2) if d == 0 else 1 - (1 - d * x / 2)**(1 / d)
    if abs(f - 1) < ptol:
        return math.exp(x / 2) if b == 0 else (1 + b * x / 2)**(1 / b)
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


_genlambda_cdf = np.vectorize(
    _genlambda_cdf0,
    [float],
    excluded={'ptol', 'xtol', 'maxiter'},
)

def _genlambda_lmo0(
    r: int,
    s: float,
    t: float,
    b: float,
    d: float,
    f: float,
) -> float:
    if r == 0:
        return 1

    if b <= -1 - s and d <= -1 - t:
        return math.nan

    def _lmo0_partial(trim: float, theta: float) -> float:
        if r == 1 and theta == 0:
            return cast(float, harmonic(trim) - harmonic(s + t + 1))

        return (
            (-1)**r *
            sc.poch(r + trim, s + t - trim + 1)  # type: ignore
            * sc.poch(1 - theta, r - 2)  # type: ignore
            / sc.poch(1 + theta + trim, r + s + t - trim)  # type: ignore
            - (1 / theta if r == 1 else 0)
        ) / r

    return (
        (1 + f) * _lmo0_partial(s, b)
        + (-1)**r * (1 - f) * _lmo0_partial(t, d)
    )

_genlambda_lmo = np.vectorize(_genlambda_lmo0, [float], excluded={1, 2})

class genlambda_gen(_rv_continuous):  # noqa: N801
    def _argcheck(self, b: float, d: float, f: float) -> int:
        return np.isfinite(b) & np.isfinite(d) & (f >= -1) & (f <= 1)

    def _shape_info(self) -> Sequence[_ShapeInfo]:
        ibeta = _ShapeInfo('b', False, (-np.inf, np.inf), (False, False))
        idelta = _ShapeInfo('d', False, (-np.inf, np.inf), (False, False))
        iphi = _ShapeInfo('f', False, (-1, 1), (True, True))
        return [ibeta, idelta, iphi]

    def _get_support(
        self,
        b: float,
        d: float,
        f: float,
    ) -> tuple[float, float]:
        return _genlambda_support(b, d, f)

    def _fitstart(
        self,
        data: npt.NDArray[np.float64],
        args: tuple[float, float, float] | None = None,
    ) -> tuple[float, float, float, float, float]:
        #  Arbitrary, but the default f=1 is a bad start
        return super()._fitstart(data, args or (1., 1., 0.))  # type: ignore

    def _pdf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return 1 / _genlambda_qdf(self._cdf(x, b, d, f), b, d, f)

    def _cdf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return _genlambda_cdf(x, b, d, f)

    def _ppf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return _genlambda_ppf(x, b, d, f)

    def _stats(self, b: float, d: float, f: float) -> tuple[
        float,
        float,
        float | None,
        float | None,
    ]:
        if b <= -1 or d <= -1:
            # hard NaN (not inf); indeterminate integral
            return math.nan, math.nan, math.nan, math.nan

        a, c = 1 + f, 1 - f
        b1, d1 = 1 + b, 1 + d

        m1 = 0 if b == d and f == 0 else _genlambda_lmo0(1, 0, 0, b, d, f)

        if b <= -1 / 2 or d <= -1 / 2:
            return m1, math.nan, math.nan, math.nan

        if b == d == 0:
            m2 = 4 * f**2 + math.pi**2 * (1 - f**2) / 3
        elif b == 0:
            m2 = (
                a**2
                + (c / d1)**2 / (d1 + d)
                + 2 * a * c / (d * d1) * (1 - cast(float, harmonic(1 + d)))
            )
        elif d == 0:
            m2 = (
                c**2
                + (a / b1)**2 / (b1 + b)
                + 2 * a * c / (b * b1) * (1 - cast(float, harmonic(1 + b)))
            )
        else:
            m2 = (
                (a / b1)**2 / (b1 + b)
                + (c / d1)**2 / (d1 + d)
                + 2 * a * c  / (b * d) * (
                    1 / (b1 * d1)
                    - cast(float, sc.beta(b1, d1))  # type: ignore
                )
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
        r: npt.NDArray[np.int64],
        b: float,
        d: float,
        f: float,
        trim: tuple[int, int] | tuple[float, float],
        quad_opts: QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim

        if quad_opts is not None:
            # only do numerical integration when quad_opts is passed
            lmbda_r = cast(
                float | npt.NDArray[np.float64],
                l_moment_from_ppf(
                    functools.partial(self._ppf, b=b, d=d, f=f), # type: ignore
                    r,
                    trim=trim,
                    quad_opts=quad_opts,
                ),  # type: ignore
            )
            return np.asarray(lmbda_r)

        return np.atleast_1d(
            cast(_ArrF8, _genlambda_lmo(r, s, t, b, d, f)),
        )


genlambda: RVContinuous[float, float, float] = genlambda_gen(
    name='genlambda',
)  # type: ignore
r"""A generalized Tukey-Lambda random variable.

`genlambda` takes `b`, `d` and `f` as shape parameters.
`b` and `d` can be any float, and `f` requires `-1 <= f <= 1`.

If `f == 0` and `b == d`, `genlambda` is equivalent to
[`scipy.stats.tukeylambda`][scipy.stats.tukeylambda], with `b` (or `d`) as
shape parameter.

For a detailed description of the GLD, refer to
[Distributions - GLD](distributions.md#gld).
"""
