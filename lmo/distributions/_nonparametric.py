from __future__ import annotations

import functools
import math
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc
import optype.typing as opt

import lmo.typing as lmt
from lmo._utils import clean_trim, l_stats_orders, moments_to_ratio, round0
from lmo.theoretical import (
    cdf_from_ppf,
    entropy_from_qdf,
    l_moment_from_ppf,
    ppf_from_l_moments,
    qdf_from_l_moments,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["l_poly"]

###

_Float1D: TypeAlias = onp.Array1D[npc.floating]
_FloatND: TypeAlias = onp.ArrayND[npc.floating]

_T = TypeVar("_T")


_Trim: TypeAlias = tuple[int, int] | tuple[float, float]
_MomentType: TypeAlias = Literal[0, 1]
_LPolyParams: TypeAlias = tuple[onp.ToFloat1D] | tuple[onp.ToFloat1D, lmt.ToTrim]
_ToShape: TypeAlias = op.CanIndex | tuple[op.CanIndex, ...]

_Stats0: TypeAlias = Literal[""]
_Stats1: TypeAlias = Literal["m", "v", "s", "k"]
_Stats2: TypeAlias = Literal["mv", "ms", "mk", "vs", "vk", "sk"]
_Stats3: TypeAlias = Literal["mvs", "mvk", "msk", "vsk"]
_Stats4: TypeAlias = Literal["mvsk"]
_Stats: TypeAlias = Literal[_Stats0, _Stats1, _Stats2, _Stats3, _Stats4]


_Tuple1: TypeAlias = tuple[_T]
_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]
_Tuple4: TypeAlias = tuple[_T, _T, _T, _T]
_Tuple4m: TypeAlias = tuple[()] | _Tuple1[_T] | _Tuple2[_T] | _Tuple3[_T] | _Tuple4[_T]


class _Fn1(Protocol):
    @overload
    def __call__(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /) -> _FloatND: ...


###


def _get_rng(s: lmt.Seed | None = None, /) -> np.random.Generator:
    return s if isinstance(s, np.random.Generator) else np.random.default_rng(s)


class l_poly:  # noqa: N801
    """
    Polynomial quantile distribution with (only) the given L-moments.
    """

    name: ClassVar[Literal["l_poly"]] = "l_poly"
    badvalue: ClassVar[float] = np.nan
    moment_type: ClassVar[_MomentType] = 1
    numargs: ClassVar[Literal[2]] = 2
    shapes: ClassVar[Literal["lmbda, trim"]] = "lmbda, trim"

    _l_moments: Final[_Float1D]
    _trim: Final[_Trim]
    _support: Final[_Tuple2[float]]

    _cdf: Final[_Fn1]
    _ppf: Final[_Fn1]
    _qdf: Final[_Fn1]

    _random_state: np.random.Generator

    def __init__(
        self,
        lmbda: onp.ToFloat1D,
        /,
        trim: lmt.ToTrim = 0,
        *,
        seed: lmt.Seed | None = None,
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
        lmbda_ = np.asarray(lmbda, dtype=np.float64).reshape(-1)

        if (_n := len(lmbda_)) < 2:
            msg = f"at least 2 L-moments required, got len(lmbda) = {_n}"
            raise ValueError(msg)
        self._l_moments = lmbda_

        self._trim = trim_ = clean_trim(trim)

        self._ppf = ppf_from_l_moments(lmbda_, trim=trim_)
        self._qdf = qdf_from_l_moments(lmbda_, trim=trim_, validate=False)

        a, b = self._ppf(np.array([0, 1]))
        self._support = float(a), float(b)

        self._cdf_single = cdf_from_ppf(self._ppf)
        self._cdf = np.vectorize(self._cdf_single, [float])

        self._random_state = _get_rng(seed)

        super().__init__()

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
    def random_state(self, seed: lmt.Seed, /) -> None:  # pyright: ignore[reportPropertyTypeMismatch]
        self._random_state = _get_rng(seed)

    @classmethod
    def fit(
        cls,
        data: onp.ToFloatND,
        /,
        moments: int | None = None,
        trim: lmt.ToTrim = 0,
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
            msg = "expected 1-d data, got shape {{x,shape}}"
            raise TypeError(msg)

        n = len(x)
        if n < 2 or np.all(x == x[0]):
            msg = f"expected at least two unique samples, got {min(n, 1)}"
            raise ValueError(msg)

        r_max = round(np.clip(np.cbrt(n), 2, 128)) if moments is None else moments

        if r_max < 2:
            msg = f"expected >1 moments, got {moments}"
            raise ValueError(msg)

        from lmo._lm import l_moment

        l_r = l_moment(x, np.arange(1, r_max + 1), trim=trim)
        return cls(l_r, trim=trim)

    @overload
    def rvs(
        self,
        /,
        size: tuple[()] | None = None,
        *,
        rng: lmt.Seed = None,
    ) -> float: ...
    @overload
    def rvs(self, /, size: _ToShape, *, rng: lmt.Seed = None) -> _FloatND: ...
    def rvs(
        self,
        /,
        size: _ToShape | None = None,
        *,
        rng: lmt.Seed = None,
    ) -> float | _FloatND:
        """
        Draw random variates from the relevant distribution.

        Args:
            size:
                Defining number of random variates, defaults to 1.
            rng:
                RNG to pass to [`numpy.random.default_rng`][numpy.random.default_rng].
                Defaults to `l_poly.random_state`.

        Returns:
            A scalar or array with shape like `size`.
        """
        rng = self._random_state if rng is None else np.random.default_rng(rng)
        return self._ppf(rng.uniform(size=size))

    @overload
    def ppf(self, p: onp.ToFloat, /) -> float: ...
    @overload
    def ppf(self, p: onp.ToFloatND, /) -> _FloatND: ...
    def ppf(self, p: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        r"""
        [Percent point function](https://w.wiki/8cQU) \( Q(p) \) (inverse of
        [CDF][lmo.distributions.l_poly.cdf], a.k.a. the quantile function) at
        \( p \) of the given distribution.

        Args:
            p: Scalar or array-like of lower tail probability values in \( [0, 1] \).

        See Also:
            - [`ppf_from_l_moments`][lmo.theoretical.ppf_from_l_moments]
        """
        return self._ppf(p)

    @overload
    def isf(self, q: onp.ToFloat, /) -> float: ...
    @overload
    def isf(self, q: onp.ToFloatND, /) -> _FloatND: ...
    def isf(self, q: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        r"""
        Inverse survival function \( \bar{Q}(q) = Q(1 - q) \) (inverse of
        [`sf`][lmo.distributions.l_poly.sf]) at \( q \).

        Args:
            q: Scalar or array-like of upper tail probability values in \( [0, 1] \).
        """
        p = 1 - np.asarray(q, np.float64)
        if p.ndim == 0 and np.isscalar(q):
            p = p[()]
        return self._ppf(p)

    @overload
    def qdf(self, p: onp.ToFloat, /) -> float: ...
    @overload
    def qdf(self, p: onp.ToFloatND, /) -> _FloatND: ...
    def qdf(self, p: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        r"""
        Quantile density function \( q \equiv \frac{\dd{Q}}{\dd{p}} \) (
        derivative of the [PPF][lmo.distributions.l_poly.ppf]) at \( p \) of
        the given distribution.

        Args:
            p: Scalar or array-like of lower tail probability values in \( [0, 1] \).

        See Also:
            - [`qdf_from_l_moments`][lmo.theoretical.ppf_from_l_moments]
        """
        return self._qdf(p)

    @overload
    def cdf(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def cdf(self, x: onp.ToFloatND, /) -> _FloatND: ...
    def cdf(self, x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
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
    def logcdf(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def logcdf(self, x: onp.ToFloatND, /) -> _FloatND: ...
    @np.errstate(divide="ignore")
    def logcdf(self, x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        r"""
        Logarithm of the cumulative distribution function (CDF) at \( x \),
        i.e. \( \ln F(x) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return np.log(self._cdf(x))

    @overload
    def sf(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def sf(self, x: onp.ToFloatND, /) -> _FloatND: ...
    def sf(self, x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        r"""
        Survival function \(S(x) = \mathrm{P}(X > x) =
        1 - \mathrm{P}(X \le x) = 1 - F(x) \) (the complement of the
        [CDF][lmo.distributions.l_poly.cdf]).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return 1 - self._cdf(x)

    @overload
    def logsf(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def logsf(self, x: onp.ToFloatND, /) -> _FloatND: ...
    @np.errstate(divide="ignore")
    def logsf(self, x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        r"""
        Logarithm of the survical function (SF) at \( x \), i.e.
        \( \ln \left( S(x) \right) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return np.log(self._cdf(x))

    @overload
    def pdf(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def pdf(self, x: onp.ToFloatND, /) -> _FloatND: ...
    def pdf(self, x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
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
    def logpdf(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def logpdf(self, x: onp.ToFloatND, /) -> _FloatND: ...
    def logpdf(self, x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        """Logarithm of the PDF."""
        return -np.log(self._qdf(self._cdf(x)))

    @overload
    def hf(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def hf(self, x: onp.ToFloatND, /) -> _FloatND: ...
    def hf(self, x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
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
        return self._ppf(0.5)

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
        s = math.sqrt(ss)
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
        return math.sqrt(self._var)

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

    def support(self, /) -> _Tuple2[float]:
        r"""
        The support \( (Q(0), Q(1)) \) of the distribution, where \( Q(p) \)
        is the [PPF][lmo.distributions.l_poly.ppf].
        """
        return self._support

    @overload
    def interval(self, confidence: onp.ToFloat, /) -> _Tuple2[float]: ...
    @overload
    def interval(self, confidence: onp.ToFloatND, /) -> _Tuple2[_FloatND]: ...
    def interval(
        self,
        confidence: onp.ToFloat | onp.ToFloatND,
        /,
    ) -> _Tuple2[float] | _Tuple2[_FloatND]:
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
            msg = "confidence must be between 0 and 1 inclusive"
            raise ValueError(msg)

        return self._ppf((1 - alpha) / 2), self._ppf((1 + alpha) / 2)

    def moment(self, n: opt.AnyInt, /) -> float:
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
        n = int(n)
        if n < 0:
            msg = f"expected n >= 0, got {n}"
            raise ValueError(msg)
        if n == 0:
            return 1

        def _integrand(u: float, /) -> float:
            return self._ppf(u) ** n

        from scipy.integrate import quad

        return quad(_integrand, 0, 1)[0]

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
    def stats(self, /, moments: _Stats = "mv") -> _Tuple4m[float]:
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

        moments_ = set(moments)
        if "m" in moments_:
            out.append(self._mean)
        if "v" in moments_:
            out.append(self._var)
        if "s" in moments_:
            out.append(self._skew)
        if "k" in moments_:
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

        def i(u: float, /) -> float:
            # the cast is safe, since `_F8 <: float` (at runtime)
            return g(ppf(u))

        from scipy.integrate import quad

        a, b, c, d = 0, 0.05, 0.95, 1
        return quad(i, a, b)[0] + quad(i, b, c)[0] + quad(i, c, d)[0]

    @overload
    def l_moment(
        self,
        r: lmt.ToOrder0D,
        /,
        trim: lmt.ToTrim | None = None,
    ) -> float: ...
    @overload
    def l_moment(
        self,
        r: lmt.ToOrderND,
        /,
        trim: lmt.ToTrim | None = None,
    ) -> _FloatND: ...
    def l_moment(
        self,
        r: lmt.ToOrder,
        /,
        trim: lmt.ToTrim | None = None,
    ) -> float | _FloatND:
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
        trim_ = self._trim if trim is None else clean_trim(trim)
        return l_moment_from_ppf(self._ppf, r, trim=trim_)

    @overload
    def l_ratio(
        self,
        r: lmt.ToOrder0D,
        k: lmt.ToOrder0D,
        /,
        trim: lmt.ToTrim | None = None,
    ) -> float: ...
    @overload
    def l_ratio(
        self,
        r: lmt.ToOrderND,
        k: lmt.ToOrder,
        /,
        trim: lmt.ToTrim | None = None,
    ) -> _FloatND: ...
    @overload
    def l_ratio(
        self,
        r: lmt.ToOrder,
        k: lmt.ToOrderND,
        /,
        trim: lmt.ToTrim | None = None,
    ) -> _FloatND: ...
    def l_ratio(
        self,
        r: lmt.ToOrder,
        k: lmt.ToOrder,
        /,
        trim: lmt.ToTrim | None = None,
    ) -> float | _FloatND:
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

    def l_stats(self, /, trim: lmt.ToTrim | None = None, moments: int = 4) -> _Float1D:
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

    def l_loc(self, /, trim: lmt.ToTrim | None = None) -> float:
        """
        L-location of the distribution, i.e. the 1st L-moment.

        Alias for `l_poly.l_moment(1, ...)`.

        See Also:
            - [`l_poly.l_moment`][lmo.distributions.l_poly.l_moment]
        """
        return self.l_moment(1, trim=trim)

    def l_scale(self, /, trim: lmt.ToTrim | None = None) -> float:
        """
        L-scale of the distribution, i.e. the 2nd L-moment.

        Alias for `l_poly.l_moment(2, ...)`.

        See Also:
            - [`l_poly.l_moment`][lmo.distributions.l_poly.l_moment]
        """
        return self.l_moment(2, trim=trim)

    def l_skew(self, /, trim: lmt.ToTrim | None = None) -> float:
        """L-skewness coefficient of the distribution; the 3rd L-moment ratio.

        Alias for `l_poly.l_ratio(3, 2, ...)`.

        See Also:
            - [`l_poly.l_ratio`][lmo.distributions.l_poly.l_ratio]
        """
        return self.l_ratio(3, 2, trim=trim)

    def l_kurtosis(self, /, trim: lmt.ToTrim | None = None) -> float:
        """L-kurtosis coefficient of the distribution; the 4th L-moment ratio.

        Alias for `l_poly.l_ratio(4, 2, ...)`.

        See Also:
            - [`l_poly.l_ratio`][lmo.distributions.l_poly.l_ratio]
        """
        return self.l_ratio(4, 2, trim=trim)

    l_kurt = l_kurtosis

    # `rv_continuous` and `rv_frozen` compatibility

    @property
    def dist(self, /) -> type[Self]:
        return type(self)

    @property
    def args(self, /) -> tuple[_Float1D, _Trim]:
        return self._l_moments, self._trim

    @property
    def kwds(self, /) -> dict[str, object]:
        return {}

    @classmethod
    def freeze(
        cls,
        lmbda: onp.ToFloat1D,
        /,
        trim: lmt.ToTrim = 0,
        **kwds: Any,
    ) -> Self:
        return cls(lmbda, trim, **kwds)

    @classmethod
    def nnlf(cls, /, theta: _LPolyParams, x: _FloatND) -> float | _FloatND:
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
