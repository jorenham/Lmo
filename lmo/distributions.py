# pyright: reportIncompatibleMethodOverride=false, reportImplicitOverride=false
# ruff: noqa: N801, PLR2004

"""
Probability distributions, compatible with [`scipy.stats`][scipy.stats].
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt
import scipy.special as sc
from scipy.integrate import quad  # pyright: ignore[reportUnknownVariableType]
from scipy.stats._distn_infrastructure import (
    _ShapeInfo,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from scipy.stats.distributions import rv_continuous as _rv_continuous

import lmo.typing as lmt
import lmo.typing.np as lnpt
import lmo.typing.scipy as lspt
from ._utils import clean_trim, l_stats_orders, moments_to_ratio, round0
from .special import harmonic
from .theoretical import (
    cdf_from_ppf,
    entropy_from_qdf,
    l_moment_from_ppf,
    ppf_from_l_moments,
    qdf_from_l_moments,
)


if TYPE_CHECKING:
    from typing_extensions import LiteralString, Self


__all__ = 'l_poly', 'kumaraswamy', 'wakeby', 'genlambda'


_ArrF8: TypeAlias = npt.NDArray[np.float64]
_T_f8 = TypeVar('_T_f8', bound=float | np.float64 | _ArrF8)

_AnyReal0D: TypeAlias = lnpt.AnyScalarInt | lnpt.AnyScalarFloat
_AnyReal1D: TypeAlias = lnpt.AnyVectorInt | lnpt.AnyVectorFloat
_AnyReal2D: TypeAlias = lnpt.AnyMatrixInt | lnpt.AnyMatrixFloat
_AnyRealND: TypeAlias = lnpt.AnyArrayInt | lnpt.AnyArrayFloat
_AnyReal3D_: TypeAlias = lnpt.AnyTensorInt | lnpt.AnyTensorFloat
_AnyReal2D_: TypeAlias = _AnyReal2D | _AnyReal3D_

_Stats0: TypeAlias = Literal['']
_Stats1: TypeAlias = Literal['m', 'v', 's', 'k']
_Stats2: TypeAlias = Literal['mv', 'ms', 'mk', 'vs', 'vk', 'sk']
_Stats3: TypeAlias = Literal['mvs', 'mvk', 'msk', 'vsk']
_Stats4: TypeAlias = Literal['mvsk']
_Stats: TypeAlias = Literal[_Stats0, _Stats1, _Stats2, _Stats3, _Stats4]

_T = TypeVar('_T')
_Tuple1: TypeAlias = tuple[_T]
_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]
_Tuple4: TypeAlias = tuple[_T, _T, _T, _T]
_Tuple4m: TypeAlias = (
    tuple[()] | _Tuple1[_T] | _Tuple2[_T] | _Tuple3[_T] | _Tuple4[_T]
)


class _VectorizedCDF(Protocol):
    @overload
    def __call__(self, x: _AnyRealND, /) -> _ArrF8: ...
    @overload
    def __call__(self, x: _AnyReal0D, /) -> float: ...


# Non-parametric

def _get_rng(seed: lnpt.Seed | None = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


_LPolyParams: TypeAlias = (
    tuple[lnpt.AnyVectorFloat] | tuple[lnpt.AnyVectorFloat, lmt.AnyTrim]
)


class l_poly:
    """
    Polynomial quantile distribution with (only) the given L-moments.
    """
    name: ClassVar[LiteralString] = 'l_poly'
    badvalue: ClassVar[float] = np.nan
    moment_type: ClassVar[Literal[0, 1]] = 1
    numargs: ClassVar[int] = 2
    shapes: ClassVar[LiteralString | None] = 'lmbda, trim'

    _l_moments: Final[_ArrF8]
    _trim: Final[tuple[float, float] | tuple[int, int]]
    _support: Final[tuple[float, float]]

    _cdf: Final[_VectorizedCDF]

    _random_state: np.random.Generator

    def __init__(
        self,
        lmbda: lnpt.AnyVectorFloat,
        /,
        trim: lmt.AnyTrim = 0,
        *,
        seed: lnpt.Seed | None = None,
    ) -> None:
        r"""
        Create a new `l_poly` instance.

        Args:
            lmbda:
                1-d array-like of L-moments \( \tlmoment{s,t}{r} \) for
                \( r = 1, 2, \ldots, R \). At least 2 L-moments are required.
                All remaining L-moments with \( r > R \) are considered zero.
            trim:
                The trim-length(s) of L-moments `lmbda`.
            seed:
                Random number generator.
        """
        _lmbda = np.asarray(lmbda)
        if (_n := len(_lmbda)) < 2:
            msg = f'at least 2 L-moments required, got len(lmbda) = {_n}'
            raise ValueError(msg)
        self._l_moments = _lmbda

        self._trim = _trim = clean_trim(trim)

        self._ppf = ppf_from_l_moments(_lmbda, trim=_trim)
        self._qdf = qdf_from_l_moments(_lmbda, trim=_trim, validate=False)

        a, b = self._ppf([0, 1])
        self._support = a, b

        self._cdf_single = cdf_from_ppf(self._ppf)
        self._cdf = np.vectorize(self._cdf_single, [float])

        self._random_state = _get_rng(seed)

    @property
    def a(self, /) -> float:
        """Lower bound of the support."""
        return self._support[0]

    @property
    def b(self, /) -> float:
        """Upper bound of the support."""
        return self._support[1]

    @property
    def random_state(self, /) -> np.random.Generator:
        """The random number generator of the distribution."""
        return self._random_state

    @random_state.setter
    def random_state(
        self,
        seed: lnpt.Seed,  # pyright: ignore[reportPropertyTypeMismatch]
        /,
    ) -> None:
        self._random_state = _get_rng(seed)

    @classmethod
    def fit(
        cls,
        data: _AnyRealND,
        /,
        moments: int | None = None,
        trim: lmt.AnyTrim = 0,
    ) -> Self:
        r"""
        Fit distribution using the (trimmed) L-moment estimates of the given
        data.

        Args:
            data:
                1-d array-like with sample observations.
            moments:
                How many sample L-moments to use, `2 <= moments < len(data)`.
                Defaults to $\sqrt[3]{n}$, where $n$ is `len(data)`.
            trim:
                The left and right trim-lengths $(s, t)$ to use. Defaults
                to $(0, 0)$.

        Returns:
            A fitted `l_poly` instance.

        Raises:
            TypeError: Invalid `data` shape.
            ValueError: Not enough `moments`.
            ValueError: If the L-moments of the data do not result in strictly
                monotinically increasing quantile function (PPF).

                This generally means that either the left, the right, or both
                `trim`-orders are too small.
        """
        x = np.asarray_chkfinite(data)
        if x.ndim != 1:
            msg = 'expected 1-d data, got shape {{x,shape}}'
            raise TypeError(msg)

        n = len(x)
        if n < 2 or np.all(x == x[0]):
            msg = f'expected at least two unique samples, got {min(n, 1)}'
            raise ValueError(msg)

        if moments is None:
            r_max = round(np.clip(np.cbrt(n), 2, 128))
        else:
            r_max = moments

        if r_max < 2:
            msg = f'expected >1 moments, got {moments}'
            raise ValueError(msg)

        from ._lm import l_moment

        l_r = l_moment(x, np.arange(1, r_max + 1), trim=trim)
        return cls(l_r, trim=trim)

    @overload
    def rvs(
        self,
        /,
        size: Literal[1] | None = ...,
        random_state: lnpt.Seed | None = ...,
    ) -> float: ...
    @overload
    def rvs(
        self,
        /,
        size: int | tuple[int, ...],
        random_state: lnpt.Seed | None = ...,
    ) -> _ArrF8: ...
    def rvs(
        self,
        /,
        size: int | tuple[int, ...] | None = None,
        random_state: lnpt.Seed | None = None,
    ) -> float | _ArrF8:
        """
        Draw random variates from the relevant distribution.

        Args:
            size:
                Defining number of random variates, defaults to 1.
            random_state:
                Seed or [`numpy.random.Generator`][numpy.random.Generator]
                instance. Defaults to
                [`l_poly.random_state`][lmo.distributions.l_poly.random_state].

        Returns:
            A scalar or array with shape like `size`.
        """
        if random_state is None:
            rng = self._random_state
        else:
            rng = np.random.default_rng(random_state)

        return self._ppf(rng.uniform(size=size))

    @overload
    def ppf(self, p: _AnyReal0D, /) -> float: ...
    @overload
    def ppf(self, p: _AnyRealND, /) -> _ArrF8: ...
    def ppf(self, p: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        r"""
        [Percent point function](https://w.wiki/8cQU) \( Q(p) \) (inverse of
        [CDF][lmo.distributions.l_poly.cdf], a.k.a. the quantile function) at
        \( p \) of the given distribution.

        Args:
            p:
                Scalar or array-like of lower tail probability values in
                \( [0, 1] \).

        See Also:
            - [`ppf_from_l_moments`][lmo.theoretical.ppf_from_l_moments]
        """
        return self._ppf(p)

    @overload
    def isf(self, q: _AnyReal0D, /) -> float: ...
    @overload
    def isf(self, q: _AnyRealND, /) -> _ArrF8: ...
    def isf(self, q: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        r"""
        Inverse survival function \( \bar{Q}(q) = Q(1 - q) \) (inverse of
        [`sf`][lmo.distributions.l_poly.sf]) at \( q \).

        Args:
            q:
                Scalar or array-like of upper tail probability values in
                \( [0, 1] \).
        """
        p = 1 - np.asarray(q)
        return self._ppf(p[()] if np.isscalar(q) else p)

    @overload
    def qdf(self, p: _AnyReal0D, /) -> float: ...
    @overload
    def qdf(self, p: _AnyRealND, /) -> _ArrF8: ...
    def qdf(self, p: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        r"""
        Quantile density function \( q \equiv \frac{\dd{Q}}{\dd{p}} \) (
        derivative of the [PPF][lmo.distributions.l_poly.ppf]) at \( p \) of
        the given distribution.

        Args:
            p:
                Scalar or array-like of lower tail probability values in
                \( [0, 1] \).

        See Also:
            - [`qdf_from_l_moments`][lmo.theoretical.ppf_from_l_moments]
        """
        return self._qdf(p)

    @overload
    def cdf(self, x: _AnyReal0D, /) -> float: ...
    @overload
    def cdf(self, x: _AnyRealND, /) -> _ArrF8: ...
    def cdf(self, x: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        r"""
        [Cumulative distribution function](https://w.wiki/3ota)
        \( F(x) = \mathrm{P}(X \le x) \) at \( x \) of the given distribution.

        Note:
            Because the CDF of `l_poly` is not analytically expressible, it
            is evaluated numerically using a root-finding algorithm.

        Args:
            x: Scalar or array-like of quantiles.
        """
        return self._cdf(x)

    @overload
    def logcdf(self, x: _AnyReal0D, /) -> float: ...
    @overload
    def logcdf(self, x: _AnyRealND, /) -> _ArrF8: ...
    @np.errstate(divide='ignore')
    def logcdf(self, x: _AnyReal0D | _AnyRealND) -> float | _ArrF8:
        r"""
        Logarithm of the cumulative distribution function (CDF) at \( x \),
        i.e. \( \ln F(x) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return np.log(self._cdf(x))

    @overload
    def sf(self, x: _AnyReal0D, /) -> float: ...
    @overload
    def sf(self, x: _AnyRealND, /) -> _ArrF8: ...

    def sf(self, x: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        r"""
        Survival function \(S(x) = \mathrm{P}(X > x) =
        1 - \mathrm{P}(X \le x) = 1 - F(x) \) (the complement of the
        [CDF][lmo.distributions.l_poly.cdf]).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return 1 - self._cdf(x)

    @overload
    def logsf(self, x: _AnyReal0D, /) -> float: ...
    @overload
    def logsf(self, x: _AnyRealND, /) -> _ArrF8: ...
    @np.errstate(divide='ignore')
    def logsf(self, x: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        r"""
        Logarithm of the survical function (SF) at \( x \), i.e.
        \( \ln \left( S(x) \right) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return np.log(self._cdf(x))

    @overload
    def pdf(self, x: _AnyReal0D, /) -> float: ...
    @overload
    def pdf(self, x: _AnyRealND, /) -> _ArrF8: ...
    def pdf(self, x: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        r"""
        Probability density function \( f \equiv \frac{\dd{F}}{\dd{x}} \)
        (derivative of the [CDF][lmo.distributions.l_poly.cdf]) at \( x \).

        By applying the [inverse function rule](https://w.wiki/8cQS), the PDF
        can also defined using the [QDF][lmo.distributions.l_poly.qdf] as
        \(  f(x) = 1 / q\big(F(x)\big) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return 1 / self._qdf(self._cdf(x))

    @overload
    def logpdf(self, x: _AnyReal0D, /) -> float: ...
    @overload
    def logpdf(self, x: _AnyRealND, /) -> _ArrF8: ...
    def logpdf(self, x: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        """Logarithm of the PDF."""
        return -np.log(self._qdf(self._cdf(x)))

    @overload
    def hf(self, x: _AnyReal0D, /) -> float: ...
    @overload
    def hf(self, x: _AnyRealND, /) -> _ArrF8: ...
    def hf(self, x: _AnyReal0D | _AnyRealND, /) -> float | _ArrF8:
        r"""
        [Hazard function
        ](https://w.wiki/8cWL#Failure_rate_in_the_continuous_sense)
        \( h(x) = f(x) / S(x) \) at \( x \), with \( f \) and \( S \) the
        [PDF][lmo.distributions.l_poly.pdf] and
        [SF][lmo.distributions.l_poly.sf], respectively.

        Args:
            x: Scalar or array-like of quantiles.
        """
        p = self._cdf(x)
        return 1 / (self._qdf(p) * (1 - p))

    def median(self, /) -> float:
        r"""
        [Median](https://w.wiki/3oaw) (50th percentile) of the distribution.
        Alias for `ppf(1 / 2)`.

        See Also:
            - [`l_poly.ppf`][lmo.distributions.l_poly.ppf]
        """
        return float(self._ppf(.5))

    @functools.cached_property
    def _mean(self, /) -> float:
        """Mean; 1st raw product-moment."""
        return self.moment(1)

    @functools.cached_property
    def _var(self, /) -> float:
        """Variance; 2nd central product-moment."""
        if not np.isfinite(m1 := self._mean):
            return np.nan

        m1_2 = m1 * m1
        m2 = self.moment(2)

        if m2 <= m1_2 or np.isnan(m2):
            return np.nan

        return m2 - m1_2

    @functools.cached_property
    def _skew(self, /) -> float:
        """Skewness; 3rd standardized central product-moment."""
        if np.isnan(ss := self._var) or ss <= 0:
            return np.nan
        if np.isnan(m3 := self.moment(3)):
            return np.nan

        m = self._mean
        s = np.sqrt(ss)
        u = m / s

        return m3 / s**3 - u**3 - 3 * u

    @functools.cached_property
    def _kurtosis(self, /) -> float:
        """Ex. kurtosis; 4th standardized central product-moment minus 3."""
        if np.isnan(ss := self._var) or ss <= 0:
            return np.nan
        if np.isnan(m3 := self.moment(3)):
            return np.nan
        if np.isnan(m4 := self.moment(4)):
            return np.nan

        m1 = self._mean
        uu = m1**2 / ss

        return (m4 - 4 * m1 * m3) / ss**2 + 6 * uu + 3 * uu**2 - 3

    def mean(self, /) -> float:
        r"""
        The [mean](https://w.wiki/8cQe) \( \mu = \E[X] \) of random varianble
        \( X \) of the relevant distribution.

        See Also:
            - [`l_poly.l_loc`][lmo.distributions.l_poly.l_loc]
        """
        if self._trim == (0, 0):
            return self._l_moments[0]

        return self._mean

    def var(self, /) -> float:
        r"""
        The [variance](https://w.wiki/3jNh)
        \( \Var[X] = \E\bigl[(X - \E[X])^2\bigr] =
        \E\bigl[X^2\bigr] - \E[X]^2 = \sigma^2 \) (2nd central product moment)
        of random varianble \( X \) of the relevant distribution.

        See Also:
            - [`l_poly.moment`][lmo.distributions.l_poly.moment]
        """
        return self._var

    def std(self, /) -> float:
        r"""
        The [standard deviation](https://w.wiki/3hwM)
        \( \Std[X] = \sqrt{\Var[X]} = \sigma \) of random varianble \( X \) of
        the relevant distribution.

        See Also:
            - [`l_poly.l_scale`][lmo.distributions.l_poly.l_scale]
        """
        return np.sqrt(self._var)

    @functools.cached_property
    def _entropy(self, /) -> float:
        return entropy_from_qdf(self._qdf)

    def entropy(self, /) -> float:
        r"""
        [Differential entropy](https://w.wiki/8cR3) \( \mathrm{H}[X] \) of
        random varianble \( X \) of the relevant distribution.

        It is defined as

        \[
        \mathrm{H}[X]
            = \E \bigl[ -\ln f(X) \bigr]
            = -\int_{Q(0)}^{Q(1)} \ln f(x) \dd x
            = \int_0^1 \ln q(p) \dd p ,
        \]

        with \( f(x) \) the [PDF][lmo.distributions.l_poly.pdf], \( Q(p) \)
        the [PPF][lmo.distributions.l_poly.ppf], and \( q(p) = Q'(p) \) the
        [QDF][lmo.distributions.l_poly.qdf].

        See Also:
            - [`entropy_from_qdf`][lmo.theoretical.entropy_from_qdf]
        """
        return self._entropy

    def support(self, /) -> tuple[float, float]:
        r"""
        The support \( (Q(0), Q(1)) \) of the distribution, where \( Q(p) \)
        is the [PPF][lmo.distributions.l_poly.ppf].
        """
        return self._support

    @overload
    def interval(self, confidence: _AnyRealND, /) -> tuple[_ArrF8, _ArrF8]: ...
    @overload
    def interval(self, confidence: _AnyReal0D, /) -> tuple[float, float]: ...
    def interval(
        self,
        confidence: _AnyReal0D | _AnyRealND,
        /,
    ) -> tuple[float, float] | tuple[_ArrF8, _ArrF8]:
        r"""
        [Confidence interval](https://w.wiki/3kdb) with equal areas around
        the [median][lmo.distributions.l_poly.median].

        For `confidence` level \( \alpha \in [0, 1] \), this function evaluates

        \[
            \left[
                Q\left( \frac{1 - \alpha}{2} \right) ,
                Q\left( \frac{1 + \alpha}{2} \right)
            \right],
        \]

        where \( Q(p) \) is the [PPF][lmo.distributions.l_poly.ppf].

        Args:
            confidence:
                Scalar or array-like. The Probability that a random
                varianble will be drawn from the returned range.

                Each confidence value should be between 0 and 1.
        """
        alpha = np.asarray(confidence)
        if np.any((alpha > 1) | (alpha < 0)):
            msg = 'confidence must be between 0 and 1 inclusive'
            raise ValueError(msg)

        return self._ppf((1 - alpha) / 2), self._ppf((1 + alpha) / 2)

    def moment(self, n: int | np.integer[Any], /) -> float:
        r"""
        Non-central product moment \( \E[X^n] \) of \( X \) of specified
        order \( n \).

        Note:
            The product moment is evaluated using numerical integration
            ([`scipy.integrate.quad`][scipy.integrate.quad]), which cannot
            check whether the product-moment actually exists for the
            distribution, in which case an invalid result will be returned.

        Args:
            n: Order \( n \ge 0 \) of the moment.

        See Also:
            - [`l_poly.l_moment`][lmo.distributions.l_poly.l_moment]

        Todo:
            - For n>=2, attempt tot infer from `_l_moments` if the 2nd moment
                condition holds, using `diagnostics.l_moment_bounds`.
        """
        if n < 0:
            msg = f'expected n >= 0, got {n}'
            raise ValueError(msg)
        if n == 0:
            return 1.

        def _integrand(u: float) -> float:
            return self._ppf(u)**n

        return cast(float, quad(_integrand, 0, 1)[0])

    @overload
    def stats(self, /) -> _Tuple2[float]: ...
    @overload
    def stats(self, /, moments: _Stats0) -> tuple[()]: ...
    @overload
    def stats(self, /, moments: _Stats1) -> _Tuple1[float]: ...
    @overload
    def stats(self, /, moments: _Stats2) -> _Tuple2[float]: ...
    @overload
    def stats(self, /, moments: _Stats3) -> _Tuple3[float]: ...
    @overload
    def stats(self, /, moments: _Stats4) -> _Tuple4[float]: ...
    def stats(self, /, moments: _Stats = 'mv') -> _Tuple4m[float]:
        r"""
        Some product-moment statistics of the given distribution.

        Args:
            moments:
                Composed of letters `mvsk` defining which product-moment
                statistic to compute:

                `'m'`:
                :   Mean \( \mu = \E[X] \)

                `'v'`:
                :   Variance \( \sigma^2 = \E[(X - \mu)^2] \)

                `'s'`:
                :   Skewness \( \E[(X - \mu)^3] / \sigma^3 \)

                `'k'`:
                :   Ex. Kurtosis \( \E[(X - \mu)^4] / \sigma^4 - 3 \)
        """
        out: list[float] = []

        _moments = set(moments)
        if 'm' in _moments:
            out.append(self._mean)
        if 'v' in _moments:
            out.append(self._var)
        if 's' in _moments:
            out.append(self._skew)
        if 'k' in _moments:
            out.append(self._kurtosis)

        return tuple(round0(np.array(out), 1e-15))

    def expect(self, g: Callable[[float], float], /) -> float:
        r"""
        Calculate expected value of a function with respect to the
        distribution by numerical integration.

        The expected value of a function \( g(x) \) with respect to a
        random variable \( X \) is defined as

        \[
            \E\left[ g(X) \right]
                = \int_{Q(0)}^{Q(1)} g(x) f(x) \dd x
                = \int_0^1 g\big(Q(u)\big) \dd u ,
        \]

        with \( f(x) \) the [PDF][lmo.distributions.l_poly.pdf], and
        \( Q \) the [PPF][lmo.distributions.l_poly.ppf].

        Args:
            g ( (float) -> float ):
                Continuous and deterministic function
                \( g: \reals \mapsto \reals \).
        """
        ppf = self._ppf

        def i(u: float) -> float:
            return g(ppf(u))

        a = 0
        b = 0.05
        c = 1 - b
        d = 1
        return cast(
            float,
            quad(i, a, b)[0] + quad(i, b, c)[0] + quad(i, c, d)[0],
        )

    @overload
    def l_moment(
        self,
        r: lmt.AnyOrder,
        /,
        trim: lmt.AnyTrim | None = ...,
    ) -> np.float64: ...
    @overload
    def l_moment(
        self,
        r: lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim | None = ...,
    ) -> _ArrF8: ...
    def l_moment(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim | None = None,
    ) -> np.float64 | _ArrF8:
        r"""
        Evaluate the population L-moment(s) $\lambda^{(s,t)}_r$.

        Args:
            r:
                L-moment order(s), non-negative integer or array-like of
                integers.
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
        """
        _trim = self._trim if trim is None else clean_trim(trim)
        return l_moment_from_ppf(self._ppf, r, trim=_trim)

    @overload
    def l_ratio(
        self,
        r: lmt.AnyOrder,
        k: lmt.AnyOrder,
        /,
        trim: lmt.AnyTrim | None = ...,
    ) -> np.float64: ...
    @overload
    def l_ratio(
        self,
        r: lmt.AnyOrderND,
        k: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim | None = ...,
    ) -> _ArrF8: ...
    @overload
    def l_ratio(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        k: lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim | None = ...,
    ) -> _ArrF8: ...
    def l_ratio(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        k: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim | None = None,
    ) -> np.float64 | _ArrF8:
        r"""
        Evaluate the population L-moment ratio('s) $\tau^{(s,t)}_{r,k}$.

        Args:
            r:
                L-moment order(s), non-negative integer or array-like of
                integers.
            k:
                L-moment order of the denominator, e.g. 2.
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
        """
        rs = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(k)))
        lms = self.l_moment(rs, trim=trim)
        return moments_to_ratio(rs, lms)

    def l_stats(
        self,
        /,
        trim: lmt.AnyTrim | None = None,
        moments: int = 4,
    ) -> _ArrF8:
        r"""
        Evaluate the L-moments (for $r \le 2$) and L-ratio's (for $r > 2$).

        Args:
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
            moments:
                The amount of L-moments to return. Defaults to 4.
        """
        r, s = l_stats_orders(moments)
        return self.l_ratio(r, s, trim=trim)

    def l_loc(self, /, trim: lmt.AnyTrim | None = None) -> float:
        """
        L-location of the distribution, i.e. the 1st L-moment.

        Alias for `l_poly.l_moment(1, ...)`.

        See Also:
            - [`l_poly.l_moment`][lmo.distributions.l_poly.l_moment]
        """
        return float(self.l_moment(1, trim=trim))

    def l_scale(self, /, trim: lmt.AnyTrim | None = None) -> float:
        """
        L-scale of the distribution, i.e. the 2nd L-moment.

        Alias for `l_poly.l_moment(2, ...)`.

        See Also:
            - [`l_poly.l_moment`][lmo.distributions.l_poly.l_moment]
        """
        return float(self.l_moment(2, trim=trim))

    def l_skew(self, /, trim: lmt.AnyTrim | None = None) -> float:
        """L-skewness coefficient of the distribution; the 3rd L-moment ratio.

        Alias for `l_poly.l_ratio(3, 2, ...)`.

        See Also:
            - [`l_poly.l_ratio`][lmo.distributions.l_poly.l_ratio]
        """
        return float(self.l_ratio(3, 2, trim=trim))

    def l_kurtosis(self, /, trim: lmt.AnyTrim | None = None) -> float:
        """L-kurtosis coefficient of the distribution; the 4th L-moment ratio.

        Alias for `l_poly.l_ratio(4, 2, ...)`.

        See Also:
            - [`l_poly.l_ratio`][lmo.distributions.l_poly.l_ratio]
        """
        return float(self.l_ratio(4, 2, trim=trim))

    l_kurt = l_kurtosis

    # `rv_continuous` and `rv_frozen` compatibility

    @property
    def dist(self, /) -> type[Self]:  # noqa: D102
        return type(self)

    @property
    def args(self, /) -> _LPolyParams:  # noqa: D102
        return (self._l_moments, self._trim)

    @property
    def kwds(self, /) -> dict[str, Any]:  # noqa: D102
        return {}

    @classmethod
    def freeze(  # noqa: D102
        cls,
        lmbda: lnpt.AnyVectorFloat,
        /,
        trim: lmt.AnyTrim = 0,
        **kwds: Any,
    ) -> Self:
        return cls(lmbda, trim, **kwds)

    @overload
    @classmethod
    def nnlf(cls, /, theta: _LPolyParams, x: _AnyReal1D) -> float: ...
    @overload
    @classmethod
    def nnlf(cls, /, theta: _LPolyParams, x: _AnyReal2D_) -> _ArrF8: ...
    @classmethod
    def nnlf(cls, /, theta: _LPolyParams, x: _AnyRealND) -> float | _ArrF8:
        """
        Negative loglikelihood function.

        This is calculated as `-sum(log pdf(x, *theta), axis=0)`,
        where `theta` are the vector of L-moments, and optionally the trim.

        Notes:
            This is mostly for compatibility `rv_generic`, and is
            impractically slow (due to the numerical inversion of the ppf).

        Args:
            theta:
                Tuple of size 1 or 2, with the L-moments vector, and optionally
                the trim (defaults to 0).
            x:
                Array-like with observations of shape `(n)` or `(n, *ks)`.

        Returns:
            Scalar or array of shape `(*ks)` with negative loglikelihoods.
        """
        match theta:
            case (lmbda,):
                rv = cls(lmbda)
            case (lmbda, trim):
                rv = cls(lmbda, trim)
            case _ as huh:  # pyright: ignore[reportUnnecessaryComparison]
                raise TypeError(huh)

        return -np.log(rv.pdf(x)).sum(axis=0)


# Parametric


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
        * cast(_ArrF8, sc.comb(r + k - 2, r + t - 1))  # pyright: ignore[reportUnknownMemberType]
        * cast(_ArrF8, sc.comb(r + s + t, k))  # pyright: ignore[reportUnknownMemberType]
        * cast(_ArrF8, sc.beta(1 / a, 1 + k * b)) / a
    ).sum() / r


_kumaraswamy_lmo = np.vectorize(_kumaraswamy_lmo0, [float], excluded={1, 2})


@final
class kumaraswamy_gen(cast(type[lspt.AnyRV], _rv_continuous)):  # pyright: ignore[reportGeneralTypeIssues]
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

    def _qdf(
        self,
        u: npt.NDArray[np.float64],
        a: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        return (
            (1 - u)**(1 / (b - 1))
            * (1 - (1 - u)**(1 / b))**(1 / (a - 1))
            / (a * b)
        )

    def _ppf(
        self,
        u: npt.NDArray[np.float64],
        a: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        return (1 - (1 - u)**(1 / b))**(1 / a)

    def _entropy(self, a: float, b: float) -> float:
        # https://wikipedia.org/wiki/Kumaraswamy_distribution
        return (1 - 1 / b) + (1 - 1 / a) * harmonic(b) - np.log(a * b)

    def _munp(
        self,
        n: int,
        a: float,
        b: float,
    ) -> float:
        return b * cast(float, sc.beta(1 + n / a, b))

    def _l_moment(
        self,
        r: npt.NDArray[np.int64],
        a: float,
        b: float,
        trim: tuple[int, int] | tuple[float, float],
        quad_opts: lspt.QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim
        if quad_opts is not None or isinstance(s, float):
            return cast(
                _ArrF8,
                super()._l_moment(  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                    r,
                    a,
                    b,
                    trim=trim,
                    quad_opts=quad_opts,
                ),
            )

        return np.atleast_1d(cast(_ArrF8, _kumaraswamy_lmo(r, s, t, a, b)))


kumaraswamy: Final = cast(
    lspt.RVContinuous,
    kumaraswamy_gen(a=0.0, b=1.0, name='kumaraswamy'),  # pyright: ignore[reportAbstractUsage]
)
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
            w / sc.lambertw(  # pyright: ignore[reportUnknownMemberType]
                w * math.exp((1 + d * x) / f - 1),
            )
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
            return harmonic(s + t + 1) - harmonic(t)

        return scale * (
            sc.poch(r + t, s + 1)
            * sc.poch(1 - theta, r - 2)
            / sc.poch(1 + theta + t, r + s)
            + (1 / theta if r == 1 else 0)
        ) / r

    return _lmo0_partial(b, f) + _lmo0_partial(-d, 1 - f)


_wakeby_lmo = np.vectorize(_wakeby_lmo0, [float], excluded={1, 2})


class wakeby_gen(cast(type[lspt.AnyRV], _rv_continuous)):
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

        return cast(float, self.a), _wakeby_ub(b, d, f)

    def _fitstart(
        self,
        data: npt.NDArray[np.float64],
        args: tuple[float, float, float] | None = None,
    ) -> tuple[float, float, float, float, float]:
        #  Arbitrary, but the default f=1 is a bad start
        return cast(
            tuple[float, float, float, float, float],
            super()._fitstart(data, args or (1., 1., .5)),  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
        )

    def _pdf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        # application of the inverse function theorem
        return 1 / self._qdf(self._cdf(x, b, d, f), b, d, f)

    def _cdf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return 1 - _wakeby_sf(x, b, d, f)

    def _qdf(
        self,
        u: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return _wakeby_qdf(u, b, d, f)

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
        quad_opts: lspt.QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim

        if quad_opts is not None:
            # only do numerical integration when quad_opts is passed
            lmbda_r = cast(
                float | npt.NDArray[np.float64],
                l_moment_from_ppf(
                    cast(
                        Callable[[float], float],
                        functools.partial(self._ppf, b=b, d=d, f=f),
                    ),
                    r,
                    trim=trim,
                    quad_opts=quad_opts,
                ),
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
            sc.hyp2f1(1, 1 / bd, 1 + 1 / bd, -f / (1 - f)),
        )


wakeby: Final = cast(
    lspt.RVContinuous,
    wakeby_gen(a=0.0, name='wakeby'),  # pyright: ignore[reportAbstractUsage]
)
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
def _genlambda_qdf(q: _T_f8, b: float, d: float, f: float) -> _T_f8:
    return cast(_T_f8, (1 + f) * q**(b - 1) + (1 - f) * (1 - q)**(d - 1))


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
            return harmonic(trim) - harmonic(s + t + 1)

        return (
            (-1)**r *
            sc.poch(r + trim, s + t - trim + 1)
            * sc.poch(1 - theta, r - 2)
            / sc.poch(1 + theta + trim, r + s + t - trim)
            - (1 / theta if r == 1 else 0)
        ) / r

    return (
        (1 + f) * _lmo0_partial(s, b)
        + (-1)**r * (1 - f) * _lmo0_partial(t, d)
    )


_genlambda_lmo = np.vectorize(_genlambda_lmo0, [float], excluded={1, 2})


class genlambda_gen(cast(type[lspt.AnyRV], _rv_continuous)):
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
        return cast(
            tuple[float, float, float, float, float],
            super()._fitstart(data, args or (1., 1., 0.)),  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
        )

    def _pdf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return 1 / self._qdf(self._cdf(x, b, d, f), b, d, f)

    def _cdf(
        self,
        x: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return _genlambda_cdf(x, b, d, f)

    def _qdf(
        self,
        u: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return _genlambda_qdf(u, b, d, f)

    def _ppf(
        self,
        u: npt.NDArray[np.float64],
        b: float,
        d: float,
        f: float,
    ) -> npt.NDArray[np.float64]:
        return _genlambda_ppf(u, b, d, f)

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
                + 2 * a * c / (d * d1) * (1 - harmonic(1 + d))
            )
        elif d == 0:
            m2 = (
                c**2
                + (a / b1)**2 / (b1 + b)
                + 2 * a * c / (b * b1) * (1 - harmonic(1 + b))
            )
        else:
            m2 = (
                (a / b1)**2 / (b1 + b)
                + (c / d1)**2 / (d1 + d)
                + 2 * a * c / (b * d) * (
                    1 / (b1 * d1) - cast(float, sc.beta(b1, d1))
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
        quad_opts: lspt.QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim

        if quad_opts is not None:
            # only do numerical integration when quad_opts is passed
            lmbda_r = cast(
                float | npt.NDArray[np.float64],
                l_moment_from_ppf(
                    cast(
                        Callable[[float], float],
                        functools.partial(self._ppf, b=b, d=d, f=f),
                    ),
                    r,
                    trim=trim,
                    quad_opts=quad_opts,
                ),
            )
            return np.asarray(lmbda_r)

        return np.atleast_1d(
            cast(_ArrF8, _genlambda_lmo(r, s, t, b, d, f)),
        )


genlambda: Final = cast(
    lspt.RVContinuous,
    genlambda_gen(name='genlambda'),  # pyright: ignore[reportAbstractUsage]
)
r"""A generalized Tukey-Lambda random variable.

`genlambda` takes `b`, `d` and `f` as shape parameters.
`b` and `d` can be any float, and `f` requires `-1 <= f <= 1`.

If `f == 0` and `b == d`, `genlambda` is equivalent to
[`scipy.stats.tukeylambda`][scipy.stats.tukeylambda], with `b` (or `d`) as
shape parameter.

For a detailed description of the GLD, refer to
[Distributions - GLD](distributions.md#gld).
"""
