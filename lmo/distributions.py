"""Probability distributions, compatible with [`scipy.stats`][scipy.stats]."""
__all__ = (
    'l_poly',
    'l_rv_nonparametric',
    'kumaraswamy',
    'wakeby',
    'genlambda',
)

# pyright: reportIncompatibleMethodOverride=false

import functools
import math
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    Final,
    Literal,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    cast,
    overload,
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
    broadstack,
    clean_order,
    clean_trim,
    l_stats_orders,
    moments_to_ratio,
    round0,
)
from .diagnostic import l_ratio_bounds
from .special import harmonic
from .theoretical import (
    _VectorizedPPF,  # type: ignore [reportPrivateUsage]
    cdf_from_ppf,
    entropy_from_qdf,
    l_moment_from_ppf,
    ppf_from_l_moments,
    qdf_from_l_moments,
)
from .typing import (
    AnyInt,
    AnyNDArray,
    AnyScalar,
    AnyTrim,
    FloatVector,
    IntVector,
    PolySeries,
    QuadOptions,
    RVContinuous,
)

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

T = TypeVar('T')
X = TypeVar('X', bound='l_rv_nonparametric')
F = TypeVar('F', bound=np.floating[Any])
M = TypeVar('M', bound=Callable[..., Any])
V = TypeVar('V', bound=float | npt.NDArray[np.float64])

_ArrF8: TypeAlias = npt.NDArray[np.float64]

_STATS0: TypeAlias = Literal['']
_STATS1: TypeAlias = Literal['m', 'v', 's', 'k']
_STATS2: TypeAlias = Literal['mv', 'ms', 'mk', 'vs', 'vk', 'sk']
_STATS3: TypeAlias = Literal['mvs', 'mvk', 'msk', 'vsk']
_STATS4: TypeAlias = Literal['mvsk']
_STATS: TypeAlias = _STATS0 | _STATS1 | _STATS2 | _STATS3 | _STATS4

_F_EPS: Final[np.float64] = np.finfo(float).eps

# Non-parametric

class l_poly:  # noqa: N801
    """
    Polynomial quantile distribution with (only) the given L-moments.

    Todo:
        - Examples
        - `stats(moments='mv')`
    """

    _l_moments: Final[_ArrF8]
    _trim: Final[tuple[float, float] | tuple[int, int]]
    _support: Final[tuple[float, float]]

    _ppf: Final[_VectorizedPPF]
    _qdf: Final[_VectorizedPPF]
    _cdf: Final[_VectorizedPPF]

    _random_state: np.random.Generator

    def __init__(
        self,
        lmbda: npt.ArrayLike,
        /,
        trim: AnyTrim = (0, 0),
        *,
        seed: np.random.Generator | AnyInt | None = None,
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

        self._random_state = np.random.default_rng(seed)

    @property
    def random_state(self) -> np.random.Generator:
        """The random number generator of the distribution."""
        return self._random_state

    @random_state.setter
    def random_state(self, seed: int | np.random.Generator):
        self._random_state = np.random.default_rng(seed)

    @classmethod
    def fit(
        cls,
        data: npt.ArrayLike,
        moments: int | None = None,
        trim: AnyTrim = (0, 0),
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
        size: Literal[1] | None = ...,
        random_state: np.random.Generator | AnyInt | None = ...,
    ) -> float: ...

    @overload
    def rvs(
        self,
        size: int | tuple[int, ...],
        random_state: np.random.Generator | AnyInt | None = ...,
    ) -> _ArrF8: ...

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        random_state: np.random.Generator | AnyInt | None = None,
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
    def ppf(self, p: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def ppf(self, p: AnyScalar) -> float: ...
    def ppf(self, p: npt.ArrayLike) -> float | _ArrF8:
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
    def isf(self, q: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def isf(self, q: AnyScalar) -> float: ...
    def isf(self, q: npt.ArrayLike) -> float | _ArrF8:
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
    def qdf(self, p: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def qdf(self, p: AnyScalar) -> float: ...
    def qdf(self, p: npt.ArrayLike) -> float | _ArrF8:
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
    def cdf(self, x: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def cdf(self, x: AnyScalar) -> float: ...
    def cdf(self, x: npt.ArrayLike) -> float | _ArrF8:
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
    def logcdf(self, x: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def logcdf(self, x: AnyScalar) -> float: ...
    @np.errstate(divide='ignore')
    def logcdf(self, x: npt.ArrayLike) -> float | _ArrF8:
        r"""
        Logarithm of the cumulative distribution function (CDF) at \( x \),
        i.e. \( \ln F(x) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return np.log(self._cdf(x))

    @overload
    def sf(self, x: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def sf(self, x: AnyScalar) -> float: ...
    def sf(self, x: npt.ArrayLike) -> float | _ArrF8:
        r"""
        Survival function \(S(x) = \mathrm{P}(X > x) =
        1 - \mathrm{P}(X \le x) = 1 - F(x) \) (the complement of the
        [CDF][lmo.distributions.l_poly.cdf]).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return 1 - self._cdf(x)

    @overload
    def logsf(self, x: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def logsf(self, x: AnyScalar) -> float: ...
    @np.errstate(divide='ignore')
    def logsf(self, x: npt.ArrayLike) -> float | _ArrF8:
        r"""
        Logarithm of the survical function (SF) at \( x \), i.e.
        \( \ln \left( S(x) \right) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return np.log(self._cdf(x))

    @overload
    def pdf(self, x: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def pdf(self, x: AnyScalar) -> float: ...
    def pdf(self, x: npt.ArrayLike) -> float | _ArrF8:
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
    def hf(self, x: AnyNDArray[Any] | Sequence[Any]) -> _ArrF8: ...
    @overload
    def hf(self, x: AnyScalar) -> float: ...
    def hf(self, x: npt.ArrayLike) -> float | _ArrF8:
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

    def median(self) -> float:
        r"""
        [Median](https://w.wiki/3oaw) (50th percentile) of the distribution.
        Alias for `ppf(1 / 2)`.

        See Also:
            - [`l_poly.ppf`][lmo.distributions.l_poly.ppf]
        """
        return self._ppf(.5)

    @functools.cached_property
    def _mean(self) -> float:
        """Mean; 1st raw product-moment."""
        return self.moment(1)

    @functools.cached_property
    def _var(self) -> float:
        """Variance; 2nd central product-moment."""
        if not np.isfinite(m1 := self._mean):
            return np.nan

        m1_2 = m1 * m1
        m2 = self.moment(2)

        if m2 <= m1_2 or np.isnan(m2):
            return np.nan

        return m2 - m1_2

    @functools.cached_property
    def _skew(self) -> float:
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
    def _kurtosis(self) -> float:
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

    def mean(self) -> float:
        r"""
        The [mean](https://w.wiki/8cQe) \( \mu = \E[X] \) of random varianble
        \( X \) of the relevant distribution.

        See Also:
            - [`l_poly.l_loc`][lmo.distributions.l_poly.l_loc]
        """
        if self._trim == (0, 0):
            return self._l_moments[0]

        return self._mean

    def var(self) -> float:
        r"""
        The [variance](https://w.wiki/3jNh)
        \( \Var[X] = \E\bigl[(X - \E[X])^2\bigr] =
        \E\bigl[X^2\bigr] - \E[X]^2 = \sigma^2 \) (2nd central product moment)
        of random varianble \( X \) of the relevant distribution.

        See Also:
            - [`l_poly.moment`][lmo.distributions.l_poly.moment]
        """
        return self._var

    def std(self) -> float:
        r"""
        The [standard deviation](https://w.wiki/3hwM)
        \( \Std[X] = \sqrt{\Var[X]} = \sigma \) of random varianble \( X \) of
        the relevant distribution.

        See Also:
            - [`l_poly.l_scale`][lmo.distributions.l_poly.l_scale]
        """
        return np.sqrt(self._var)

    @functools.cached_property
    def _entropy(self) -> float:
        return entropy_from_qdf(self._qdf)

    def entropy(self) -> float:
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

    def support(self) -> tuple[float, float]:
        r"""
        The support \( (Q(0), Q(1)) \) of the distribution, where \( Q(p) \)
        is the [PPF][lmo.distributions.l_poly.ppf].
        """
        return self._support

    @overload
    def interval(
        self,
        confidence: AnyNDArray[Any] | Sequence[Any],
        /,
    ) -> tuple[_ArrF8, _ArrF8]: ...
    @overload
    def interval(self, confidence: AnyScalar, /) -> tuple[float, float]: ...
    def interval(
        self,
        confidence: npt.ArrayLike,
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

    def moment(self, n: float, /) -> float:
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

        from scipy.integrate import quad  # type: ignore

        return cast(float, quad(_integrand, 0, 1)[0])

    @overload
    def stats(self, moments: _STATS0) -> tuple[()]: ...
    @overload
    def stats(self, moments: _STATS1) -> tuple[float]: ...
    @overload
    def stats(self, moments: _STATS2 = ...) -> tuple[float, float]: ...
    @overload
    def stats(self, moments: _STATS3) -> tuple[float, float, float]: ...
    @overload
    def stats(self, moments: _STATS4) -> tuple[float, float, float, float]: ...

    def stats(self, moments: _STATS = 'mv') -> tuple[float, ...]:
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
        if 'm' in moments:
            out.append(self._mean)
        if 'v' in moments:
            out.append(self._var)
        if 's' in moments:
            out.append(self._skew)
        if 'k' in moments:
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

        from scipy.integrate import quad  # type: ignore

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
        r: IntVector,
        /,
        trim: AnyTrim | None = ...,
    ) -> _ArrF8: ...

    @overload
    def l_moment(
        self,
        r: AnyInt,
        /,
        trim: AnyTrim | None = ...,
    ) -> np.float64: ...

    def l_moment(
        self,
        r: AnyInt | IntVector,
        /,
        trim: AnyTrim | None = None,
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
        r: IntVector,
        k: AnyInt | IntVector,
        /,
        trim: AnyTrim | None = ...,
    ) -> _ArrF8: ...

    @overload
    def l_ratio(
        self,
        r: AnyInt | IntVector,
        k: IntVector,
        /,
        trim: AnyTrim | None = ...,
    ) -> _ArrF8: ...

    @overload
    def l_ratio(
        self,
        r: AnyInt,
        k: AnyInt,
        /,
        trim: AnyTrim | None = ...,
    ) -> np.float64: ...

    def l_ratio(
        self,
        r: AnyInt | IntVector,
        k: AnyInt | IntVector,
        /,
        trim: AnyTrim | None = None,
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
        rs = broadstack(r, k)
        lms = self.l_moment(rs, trim=trim)
        return moments_to_ratio(rs, lms)

    def l_stats(self, trim: AnyTrim | None = None, moments: int = 4) -> _ArrF8:
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

    def l_loc(self, trim: AnyTrim | None = None) -> float:
        """
        L-location of the distribution, i.e. the 1st L-moment.

        Alias for `l_poly.l_moment(1, ...)`.

        See Also:
            - [`l_poly.l_moment`][lmo.distributions.l_poly.l_moment]
        """
        return float(self.l_moment(1, trim=trim))

    def l_scale(self, trim: AnyTrim | None = None) -> float:
        """
        L-scale of the distribution, i.e. the 2nd L-moment.

        Alias for `l_poly.l_moment(2, ...)`.

        See Also:
            - [`l_poly.l_moment`][lmo.distributions.l_poly.l_moment]
        """
        return float(self.l_moment(2, trim=trim))

    def l_skew(self, trim: AnyTrim | None = None) -> float:
        """L-skewness coefficient of the distribution; the 3rd L-moment ratio.

        Alias for `l_poly.l_ratio(3, 2, ...)`.

        See Also:
            - [`l_poly.l_ratio`][lmo.distributions.l_poly.l_ratio]
        """
        return float(self.l_ratio(3, 2, trim=trim))

    def l_kurtosis(self, trim: AnyTrim | None = None) -> float:
        """L-kurtosis coefficient of the distribution; the 4th L-moment ratio.

        Alias for `l_poly.l_ratio(4, 2, ...)`.

        See Also:
            - [`l_poly.l_ratio`][lmo.distributions.l_poly.l_ratio]
        """
        return float(self.l_ratio(4, 2, trim=trim))

def _check_lmoments(
    l_r: npt.NDArray[np.floating[Any]],
    trim: AnyTrim = (0, 0),
    name: str = 'lmbda',
):
    if (n := len(l_r)) < 2:
        msg = f'at least 2 L-moments required, got {n}'
        raise ValueError(msg)
    if l_r[1] <= 0:
        msg = f'L-scale must be positive, got {name}[1] = {l_r[1]}'
    if n == 2:
        return

    r = np.arange(1, n + 1)
    t_r = l_r[2:] / l_r[1]
    t_r_max = l_ratio_bounds(r[2:], trim, legacy=True)[1]
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
    Warning:
        `l_rv_nonparametric` is deprecated, and will be removed in version
        `0.13`. Use `l_poly` instead.

    Estimate a distribution using the given L-moments.
    See [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] for the
    available method.

    The PPF (quantile function) is estimated using generalized Fourier series,
    with the (shifted) Jacobi orthogonal polynomials as basis, and the (scaled)
    L-moments as coefficients.

    The *corrected* version of theorem 3 from Hosking (2007) states that

    $$
    \widehat{Q}(u) = \sum_{r=1}^{R}
        \frac{r}{r + s + t} (2r + s + t - 1)
        \lambda^{(s, t)}_r
        \shjacobi{r - 1}{t}{s}{2u - 1} \ ,
    $$

    converges almost everywhere as \( R \rightarrow \infty \), for any
    sufficiently smooth quantile function (PPF) \( Q(u) \) on
    \( u \in (0, 1) \).
    Here, \( \shjacobi n \alpha \beta x = \jacobi{n}{\alpha}{\beta}{2x - 1} \)
    is a shifted Jacobi polynomial.

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
        - [Wolfram Research - Jacobi polynomial Fourier Expansion](
            http://functions.wolfram.com/05.06.25.0007.01)

    See Also:
        - [Jacobi Polynomials - Wikipedia](
            https://wikipedia.org/wiki/Jacobi_polynomials)
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

        self._trim = _trim = (s, t) = clean_trim(trim)

        _check_lmoments(l_r, _trim)
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
            weight function $q^s (1 - q)^t$ of quantile $q \in [0, 1]$,
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
                f'multiple fixed points at {x = :.6f}: '  # noqa: E203
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
        quad_opts: QuadOptions | None = None,
    ) -> _ArrF8:
        s, t = trim

        if quad_opts is not None:
            # only do numerical integration when quad_opts is passed
            lmbda_r = cast(
                float | npt.NDArray[np.float64],
                l_moment_from_ppf(
                    functools.partial(
                        self._ppf,
                        b=b,
                        d=d,
                        f=f,
                    ),  # type: ignore
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
                + 2 * a * c / (b * d) * (
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
                    functools.partial(
                        self._ppf,
                        b=b,
                        d=d,
                        f=f,
                    ),  # type: ignore
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
