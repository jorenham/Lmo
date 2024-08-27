from __future__ import annotations

import functools
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    TypeAlias,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad  # pyright: ignore[reportUnknownVariableType]

import lmo.typing as lmt
import lmo.typing.np as lnpt
from lmo._utils import clean_trim, l_stats_orders, moments_to_ratio, round0
from lmo.theoretical import (
    cdf_from_ppf,
    entropy_from_qdf,
    l_moment_from_ppf,
    ppf_from_l_moments,
    qdf_from_l_moments,
)


if sys.version_info >= (3, 13):
    from typing import LiteralString, Protocol, Self, TypeVar
else:
    from typing_extensions import LiteralString, Protocol, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    import optype.numpy as onpt


_Trim: TypeAlias = tuple[int, int] | tuple[float, float]
_MomentType: TypeAlias = Literal[0, 1]
_LPolyParams: TypeAlias = (
    tuple[lnpt.AnyVectorFloat]
    | tuple[lnpt.AnyVectorFloat, lmt.AnyTrim]
)


_AnyReal0D: TypeAlias = float | np.bool_ | lnpt.Int | lnpt.Float
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


NaN: Final[np.float64] = np.float64(np.nan)


class _VectorizedF(Protocol):
    @overload
    def __call__(self, x: _AnyReal0D, /) -> np.float64: ...
    @overload
    def __call__(self, x: _AnyRealND, /) -> npt.NDArray[np.float64]: ...


def _get_rng(seed: lnpt.Seed | None = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


class l_poly:  # noqa: N801
    """
    Polynomial quantile distribution with (only) the given L-moments.
    """

    name: ClassVar[LiteralString] = 'l_poly'
    badvalue: ClassVar[float] = np.nan
    moment_type: ClassVar[_MomentType] = 1
    numargs: ClassVar[int] = 2
    shapes: ClassVar[LiteralString | None] = 'lmbda, trim'

    _l_moments: Final[onpt.Array[tuple[int], np.float64]]
    _trim: Final[_Trim]
    _support: Final[tuple[np.float64, np.float64]]

    _cdf: Final[_VectorizedF]
    _ppf: Final[_VectorizedF]
    _qdf: Final[_VectorizedF]

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
    def a(self, /) -> np.float64:
        """Lower bound of the support."""
        return self._support[0]

    @property
    def b(self, /) -> np.float64:
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

        from lmo._lm import l_moment

        l_r = l_moment(x, np.arange(1, r_max + 1), trim=trim)
        return cls(l_r, trim=trim)

    @overload
    def rvs(
        self,
        /,
        size: Literal[1] | None = ...,
        random_state: lnpt.Seed | None = ...,
    ) -> np.float64: ...
    @overload
    def rvs(
        self,
        /,
        size: int | tuple[int, ...],
        random_state: lnpt.Seed | None = ...,
    ) -> npt.NDArray[np.float64]: ...
    def rvs(
        self,
        /,
        size: int | tuple[int, ...] | None = None,
        random_state: lnpt.Seed | None = None,
    ) -> np.float64 | npt.NDArray[np.float64]:
        """
        Draw random variates from the relevant distribution.

        Args:
            size:
                Defining number of random variates, defaults to 1.
            random_state:
                Seed or [`numpy.random.Generator`][numpy.random.Generator]
                instance. Defaults to `l_poly.random_state`.

        Returns:
            A scalar or array with shape like `size`.
        """
        if random_state is None:
            rng = self._random_state
        else:
            rng = np.random.default_rng(random_state)

        return self._ppf(rng.uniform(size=size))

    @overload
    def ppf(self, p: _AnyReal0D, /) -> np.float64: ...
    @overload
    def ppf(self, p: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    def ppf(
        self,
        p: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
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
    def isf(self, q: _AnyReal0D, /) -> np.float64: ...
    @overload
    def isf(self, q: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    def isf(
        self,
        q: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
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
    def qdf(self, p: _AnyReal0D, /) -> np.float64: ...
    @overload
    def qdf(self, p: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    def qdf(
        self,
        p: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
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
    def cdf(self, x: _AnyReal0D, /) -> np.float64: ...
    @overload
    def cdf(self, x: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    def cdf(
        self,
        x: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
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
    def logcdf(self, x: _AnyReal0D, /) -> np.float64: ...
    @overload
    def logcdf(self, x: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    @np.errstate(divide='ignore')
    def logcdf(
        self,
        x: _AnyReal0D | _AnyRealND,
    ) -> np.float64 | npt.NDArray[np.float64]:
        r"""
        Logarithm of the cumulative distribution function (CDF) at \( x \),
        i.e. \( \ln F(x) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return np.log(self._cdf(x))

    @overload
    def sf(self, x: _AnyReal0D, /) -> np.float64: ...
    @overload
    def sf(self, x: _AnyRealND, /) -> npt.NDArray[np.float64]: ...

    def sf(
        self,
        x: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
        r"""
        Survival function \(S(x) = \mathrm{P}(X > x) =
        1 - \mathrm{P}(X \le x) = 1 - F(x) \) (the complement of the
        [CDF][lmo.distributions.l_poly.cdf]).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return 1 - self._cdf(x)

    @overload
    def logsf(self, x: _AnyReal0D, /) -> np.float64: ...
    @overload
    def logsf(self, x: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    @np.errstate(divide='ignore')
    def logsf(
        self,
        x: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
        r"""
        Logarithm of the survical function (SF) at \( x \), i.e.
        \( \ln \left( S(x) \right) \).

        Args:
            x: Scalar or array-like of quantiles.
        """
        return np.log(self._cdf(x))

    @overload
    def pdf(self, x: _AnyReal0D, /) -> np.float64: ...
    @overload
    def pdf(self, x: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    def pdf(
        self,
        x: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
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
    def logpdf(self, x: _AnyReal0D, /) -> np.float64: ...
    @overload
    def logpdf(self, x: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    def logpdf(
        self,
        x: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
        """Logarithm of the PDF."""
        return -np.log(self._qdf(self._cdf(x)))

    @overload
    def hf(self, x: _AnyReal0D, /) -> np.float64: ...
    @overload
    def hf(self, x: _AnyRealND, /) -> npt.NDArray[np.float64]: ...
    def hf(
        self,
        x: _AnyReal0D | _AnyRealND,
        /,
    ) -> np.float64 | npt.NDArray[np.float64]:
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

    def median(self, /) -> np.float64:
        r"""
        [Median](https://w.wiki/3oaw) (50th percentile) of the distribution.
        Alias for `ppf(1 / 2)`.

        See Also:
            - [`l_poly.ppf`][lmo.distributions.l_poly.ppf]
        """
        return self._ppf(0.5)

    @functools.cached_property
    def _mean(self, /) -> np.float64:
        """Mean; 1st raw product-moment."""
        return self.moment(1)

    @functools.cached_property
    def _var(self, /) -> np.float64:
        """Variance; 2nd central product-moment."""
        if not np.isfinite(m1 := self._mean):
            return NaN

        m1_2 = m1 * m1
        m2 = self.moment(2)

        if m2 <= m1_2 or np.isnan(m2):
            return NaN

        return m2 - m1_2

    @functools.cached_property
    def _skew(self, /) -> np.float64:
        """Skewness; 3rd standardized central product-moment."""
        if np.isnan(ss := self._var) or ss <= 0:
            return NaN
        if np.isnan(m3 := self.moment(3)):
            return NaN

        m = self._mean
        s = np.sqrt(ss)
        u = m / s

        return m3 / s**3 - u**3 - 3 * u

    @functools.cached_property
    def _kurtosis(self, /) -> np.float64:
        """Ex. kurtosis; 4th standardized central product-moment minus 3."""
        if np.isnan(ss := self._var) or ss <= 0:
            return NaN
        if np.isnan(m3 := self.moment(3)):
            return NaN
        if np.isnan(m4 := self.moment(4)):
            return NaN

        m1 = self._mean
        uu = m1**2 / ss

        return (m4 - 4 * m1 * m3) / ss**2 + 6 * uu + 3 * uu**2 - 3

    def mean(self, /) -> np.float64:
        r"""
        The [mean](https://w.wiki/8cQe) \( \mu = \E[X] \) of random varianble
        \( X \) of the relevant distribution.

        See Also:
            - [`l_poly.l_loc`][lmo.distributions.l_poly.l_loc]
        """
        if self._trim == (0, 0):
            return self._l_moments[0]

        return self._mean

    def var(self, /) -> np.float64:
        r"""
        The [variance](https://w.wiki/3jNh)
        \( \Var[X] = \E\bigl[(X - \E[X])^2\bigr] =
        \E\bigl[X^2\bigr] - \E[X]^2 = \sigma^2 \) (2nd central product moment)
        of random varianble \( X \) of the relevant distribution.

        See Also:
            - [`l_poly.moment`][lmo.distributions.l_poly.moment]
        """
        return self._var

    def std(self, /) -> np.float64:
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

    def support(self, /) -> tuple[np.float64, np.float64]:
        r"""
        The support \( (Q(0), Q(1)) \) of the distribution, where \( Q(p) \)
        is the [PPF][lmo.distributions.l_poly.ppf].
        """
        return self._support

    @overload
    def interval(
        self,
        confidence: _AnyReal0D,
        /,
    ) -> tuple[np.float64, np.float64]: ...
    @overload
    def interval(
        self,
        confidence: _AnyRealND,
        /,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def interval(
        self,
        confidence: _AnyReal0D | _AnyRealND,
        /,
    ) -> (
        tuple[np.float64, np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
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

    def moment(self, n: int | np.integer[npt.NBitBase], /) -> np.float64:
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
            return np.float64(1)

        def _integrand(u: float | lnpt.Float, /) -> np.float64:
            return cast(np.float64, self._ppf(u) ** n)

        return cast(np.float64, quad(_integrand, 0, 1)[0])

    @overload
    def stats(self, /) -> _Tuple2[np.float64]: ...
    @overload
    def stats(self, /, moments: _Stats0) -> tuple[()]: ...
    @overload
    def stats(self, /, moments: _Stats1) -> _Tuple1[np.float64]: ...
    @overload
    def stats(self, /, moments: _Stats2) -> _Tuple2[np.float64]: ...
    @overload
    def stats(self, /, moments: _Stats3) -> _Tuple3[np.float64]: ...
    @overload
    def stats(self, /, moments: _Stats4) -> _Tuple4[np.float64]: ...
    def stats(self, /, moments: _Stats = 'mv') -> _Tuple4m[np.float64]:
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
        out: list[np.float64] = []

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

    def expect(
        self,
        g: Callable[[float], float | lnpt.Float],
        /,
    ) -> np.float64:
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

        def i(u: float, /) -> float | lnpt.Float:
            # the cast is safe, since `np.float64 <: float` (at runtime)
            return g(cast(float, ppf(u)))

        a = 0
        b = 0.05
        c = 1 - b
        d = 1
        return cast(
            np.float64,
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
    ) -> npt.NDArray[np.float64]: ...
    def l_moment(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim | None = None,
    ) -> np.float64 | npt.NDArray[np.float64]:
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
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def l_ratio(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        k: lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim | None = ...,
    ) -> npt.NDArray[np.float64]: ...
    def l_ratio(
        self,
        r: lmt.AnyOrder | lmt.AnyOrderND,
        k: lmt.AnyOrder | lmt.AnyOrderND,
        /,
        trim: lmt.AnyTrim | None = None,
    ) -> np.float64 | npt.NDArray[np.float64]:
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
    ) -> npt.NDArray[np.float64]:
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
    def dist(self, /) -> type[Self]:
        return type(self)

    @property
    def args(self, /) -> tuple[onpt.Array[tuple[int], np.float64], _Trim]:
        return self._l_moments, self._trim

    @property
    def kwds(self, /) -> dict[str, object]:
        return {}

    @classmethod
    def freeze(
        cls,
        lmbda: lnpt.AnyVectorFloat,
        /,
        trim: lmt.AnyTrim = 0,
        **kwds: Any,
    ) -> Self:
        return cls(lmbda, trim, **kwds)

    @overload
    @classmethod
    def nnlf(cls, /, theta: _LPolyParams, x: _AnyReal1D) -> np.float64: ...
    @overload
    @classmethod
    def nnlf(
        cls,
        /,
        theta: _LPolyParams,
        x: _AnyReal2D_,
    ) -> npt.NDArray[np.float64]: ...
    @classmethod
    def nnlf(
        cls,
        /,
        theta: _LPolyParams,
        x: _AnyRealND,
    ) -> np.float64 | npt.NDArray[np.float64]:
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
