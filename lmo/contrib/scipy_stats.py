# pyright: reportUninitializedInstanceVariable=false
"""
Extension methods for the (univariate) distributions in
[`scipy.stats`][scipy.stats].
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    TypedDict,
    Unpack,
    cast,
    overload,
)

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

import lmo.typing as lmt
from lmo import inference
from lmo._lm import l_moment as l_moment_est
from lmo._utils import clean_order, clean_trim, l_stats_orders, moments_to_ratio, round0
from lmo.distributions._lm import get_lm_func, has_lm_func, prefers_ppf
from lmo.theoretical import (
    l_moment_cov_from_cdf,
    l_moment_from_cdf,
    l_moment_from_ppf,
    l_moment_influence_from_cdf,
    l_ratio_influence_from_cdf,
    l_stats_cov_from_cdf,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from scipy.stats import rv_discrete


__all__ = ["install", "l_rv_frozen", "l_rv_generic"]

###

_FloatND: TypeAlias = onp.ArrayND[npc.floating]

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple4: TypeAlias = tuple[_T, _T, _T, _T]

_Tss = ParamSpec("_Tss")
_RVFunc: TypeAlias = lmt.Callable2[
    Callable[Concatenate[onp.ToFloat, _Tss], float],
    Callable[Concatenate[onp.ToFloatND, _Tss], _FloatND],
]


class _Fn1(Protocol):
    @overload
    def __call__(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /) -> _FloatND: ...


class _TrimKwargs(TypedDict, total=False):
    trim: tuple[int, int] | tuple[float, float]


###


class PatchClass:
    patched: ClassVar[set[type[object]]] = set()

    @classmethod
    def patch(cls, base: type[object]) -> None:
        if not isinstance(base, type):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = "patch() argument must be a type"
            raise TypeError(msg)
        if base in cls.patched:
            msg = f"{base.__qualname__} already patched"
            raise TypeError(msg)

        for name, method in cls.__dict__.items():
            if name.startswith("__") or not callable(method):
                continue
            if hasattr(base, name):
                msg = f"{base.__qualname__}.{name}() already exists"
                raise TypeError(msg)
            setattr(base, name, method)

        cls.patched.add(base)


class l_rv_generic(PatchClass):
    """
    Additional methods that are patched into
    [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] and
    [`scipy.stats.rv_discrete`][scipy.stats.rv_discrete].
    """

    _ctor_param: Mapping[str, Any]
    _parse_arg_template: str
    _random_state: np.random.RandomState | np.random.Generator
    _stats_has_moments: bool
    a: float | None
    b: float | None
    badvalue: float | None
    name: str
    numargs: int
    random_state: np.random.RandomState | np.random.Generator
    shapes: str

    _argcheck: Callable[..., int]
    _logpxf: _RVFunc[...]
    _cdf: _RVFunc[...]
    _fitstart: Callable[..., tuple[float, ...]]
    _get_support: Callable[..., _Tuple2[float]]
    _param_info: Callable[[], list[lmt.ShapeInfo]]
    _parse_args: Callable[..., tuple[tuple[Any, ...], float, float]]
    _ppf: _RVFunc[...]
    _shape_info: Callable[[], list[lmt.ShapeInfo]]
    _stats: Callable[..., _Tuple4[float | None]]
    _unpack_loc_scale: Callable[
        [onp.ToFloat1D],
        tuple[float, float, tuple[float, ...]],
    ]
    cdf: _RVFunc[...]
    fit: Callable[..., tuple[float, ...]]
    mean: Callable[..., float]
    ppf: _Fn1
    std: Callable[..., float]

    @np.errstate(divide="ignore")
    def _logqdf(self, u: _FloatND, *args: Any) -> _FloatND:
        """Overridable log quantile distribution function (QDF)."""
        return -self._logpxf(self._ppf(u, *args), *args)

    def _qdf(self, /, u: _FloatND, *args: Any) -> _FloatND:
        r"""
        Overridable quantile distribution function (QDF).

        Defaults to \( q(u) = 1 / f(Q(u)) \), with \( Q(u) \) the PPF
        (quantile function, inverse CDF), and \( f(x) \) the PDF or PMF.
        """
        return np.exp(self._logqdf(u, *args))

    @overload
    def l_moment(
        self,
        r: lmt.ToOrder0D,
        /,
        *args: Any,
        trim: lmt.ToTrim = ...,
        quad_opts: lmt.QuadOptions | None = ...,
        **kwds: Any,
    ) -> float: ...
    @overload
    def l_moment(
        self,
        r: lmt.ToOrderND,
        /,
        *args: Any,
        trim: lmt.ToTrim = ...,
        quad_opts: lmt.QuadOptions | None = ...,
        **kwds: Any,
    ) -> _FloatND: ...
    def l_moment(
        self,
        r: lmt.ToOrder,
        /,
        *args: Any,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
        **kwds: Any,
    ) -> float | _FloatND:
        r"""
        Population L-moment(s) $\lambda^{(s,t)}_r$.

        $$
        \lambda^{(s, t)}_r =
        \frac{r+s+t}{r}
        \frac{B(r,\,r+s+t)}{B(r+s,\,r+t)}
        \mathbb{E}_X \left[
            U^s
            \left(1 - U\right)^t
            \,\tilde{P}^{(t, s)}_{r-1}(U)
            \,X
        \right] \;,
        $$

        with $U = F_X(X)$ the *rank* of $X$, and $\tilde{P}^{(a,b)}_n(x)$ the
        shifted ($x \mapsto 2x-1$) Jacobi polynomial.

        Examples:
            Evaluate the population L-moments of the normally-distributed IQ
            test:

            >>> import lmo
            >>> from scipy.stats import norm
            >>> norm(100, 15).l_moment([1, 2, 3, 4]).round(6)
            array([100.      ,   8.462844,   0.      ,   1.037559])
            >>> _[1] * np.sqrt(np.pi)
            15.0000004

            Discrete distributions are also supported, e.g. the Binomial
            distribution:

            >>> from scipy.stats import binom
            >>> binom(10, 0.6).l_moment([1, 2, 3, 4]).round(6)
            array([ 6.      ,  0.862238, -0.019729,  0.096461])

        Args:
            r:
                L-moment order(s), non-negative integer or array-like of
                integers.
            *args:
                The shape parameter(s) for the distribution (see docstring
                of the instance object for more information)
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
            quad_opts:
                Optional dict of options to pass to
                [`scipy.integrate.quad`][scipy.integrate.quad].
            **kwds:
                Additional keyword arguments to pass to the distribution.

        Raises:
            TypeError: `r` is not integer-valued
            ValueError: `r` is empty or negative

        Returns:
            lmbda:
                The population L-moment(s), a scalar or float array like `r`.

        References:
            - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
                https://doi.org/10.1016/S0167-9473(02)00250-5)
            - [J.R.M. Hosking (2007) - Some theory and practical uses of
                trimmed L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

        See Also:
            - [`lmo.l_moment`][lmo.l_moment]: sample L-moment
        """
        r_ = np.asarray(clean_order(r), np.intp)
        (s, t) = trim_ = clean_trim(trim)

        shapes, loc, scale = self._parse_args(*args, **kwds)

        if s <= 0 and t <= 0:
            mean = self._stats(*shapes)[0]
            if mean is None:
                mean = self.mean(*shapes)
            if not np.isfinite(mean):
                # first moment condition not met
                return np.full(r_.shape, np.nan)[()]

        if not self._argcheck(*shapes):
            return np.full(r_.shape, np.nan)[()]

        # L-moments of the standard distribution (loc=0, scale=scale0)
        l0_r = _l_moment(self, r_, *shapes, trim=trim_, quad_opts=quad_opts)

        # shift (by loc) and scale
        shift_r: onp.ArrayND[np.float64] = loc * (r_ == 1)
        scale_r: onp.ArrayND[np.float64] = scale * (r_ > 0) + (r_ == 0)
        l_r = shift_r + scale_r * l0_r

        # round near zero values to 0
        return round0(l_r[()], tol=1e-15)

    @overload
    def l_ratio(
        self,
        order: lmt.ToOrder0D,
        order_denom: lmt.ToOrder,
        /,
        *args: Any,
        trim: lmt.ToTrim = ...,
        quad_opts: lmt.QuadOptions | None = ...,
        **kwds: Any,
    ) -> float: ...
    @overload
    def l_ratio(
        self,
        order: lmt.ToOrderND,
        order_denom: lmt.ToOrder,
        /,
        *args: Any,
        trim: lmt.ToTrim = ...,
        quad_opts: lmt.QuadOptions | None = ...,
        **kwds: Any,
    ) -> _FloatND: ...
    def l_ratio(
        self,
        r: lmt.ToOrder,
        k: lmt.ToOrder,
        /,
        *args: Any,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
        **kwds: Any,
    ) -> float | _FloatND:
        r"""
        L-moment ratio('s) $\tau^{(s,t)}_{r,k}$.

        $$
        \tau^{(s,t)}_{r,k} = \frac{\lambda^{(s,t)}_r}{\lambda^{(s,t)}_k}
        $$

        Unless explicitly specified, the r-th ($r>2$) L-ratio,
        $\tau^{(s,t)}_r$ refers to $\tau^{(s,t)}_{r, 2}$.
        Another special case is the L-variation, or the L-CV,
        $\tau^{(s,t)} = \tau^{(s,t)}_{2, 1}$. This is the L-moment analogue of
        the coefficient of variation.

        Examples:
            Evaluate the population L-CV and LL-CV (CV = coefficient of
            variation) of the standard Rayleigh distribution.

            >>> import lmo
            >>> from scipy.stats import distributions
            >>> X = distributions.rayleigh()
            >>> X.std() / X.mean()  # legacy CV
            0.5227232
            >>> X.l_ratio(2, 1)
            0.2928932
            >>> X.l_ratio(2, 1, trim=(0, 1))
            0.2752551

            And similarly, for the (discrete) Poisson distribution with rate
            parameter set to 2, the L-CF and LL-CV evaluate to:

            >>> X = distributions.poisson(2)
            >>> X.std() / X.mean()
            0.7071067
            >>> X.l_ratio(2, 1)
            0.3857527
            >>> X.l_ratio(2, 1, trim=(0, 1))
            0.4097538

            Note that (untrimmed) L-CV requires a higher (subdivision) limit in
            the integration routine, otherwise it'll complain that it didn't
            converge (enough) yet. This is because it's effectively integrating
            a non-smooth function, which is mathematically iffy, but works fine
            in this numerical application.

        Args:
            r:
                L-moment ratio order(s), non-negative integer or array-like of
                integers.
            k:
                L-moment order of the denominator, e.g. 2.
            *args:
                The shape parameter(s) for the distribution (see docstring
                of the instance object for more information)
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
            quad_opts:
                Optional dict of options to pass to
                [`scipy.integrate.quad`][scipy.integrate.quad].
            **kwds:
                Additional keyword arguments to pass to the distribution.

        See Also:
            - [`l_rv_generic.l_moment`
            ][lmo.contrib.scipy_stats.l_rv_generic.l_moment]
            - [`lmo.l_ratio`][lmo.l_ratio] - Sample L-moment ratio estimator
        """
        rk = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(k)))
        lms = self.l_moment(
            rk,
            *args,
            trim=trim,
            quad_opts=quad_opts,
            **kwds,
        )
        return moments_to_ratio(rk, lms)

    def l_stats(
        self,
        *args: Any,
        trim: lmt.ToTrim = 0,
        moments: int = 4,
        quad_opts: lmt.QuadOptions | None = None,
        **kwds: Any,
    ) -> _FloatND:
        r"""
        The L-moments (for $r \le 2$) and L-ratio's (for $r > 2$).

        By default, the first `moments = 4` population L-stats are calculated:

        - $\lambda^{(s,t)}_1$ - *L-loc*ation
        - $\lambda^{(s,t)}_2$ - *L-scale*
        - $\tau^{(s,t)}_3$ - *L-skew*ness coefficient
        - $\tau^{(s,t)}_4$ - *L-kurt*osis coefficient

        This method is equivalent to
        `X.l_ratio([1, 2, 3, 4], [0, 0, 2, 2], *, **)`, for with default
        `moments = 4`.

        Examples:
            Summarize the standard exponential distribution for different
            trim-orders.

            >>> import lmo
            >>> from scipy.stats import distributions
            >>> X = distributions.expon()
            >>> X.l_stats().round(6)
            array([1.      , 0.5     , 0.333333, 0.166667])
            >>> X.l_stats(trim=(0, 1 / 2)).round(6)
            array([0.666667, 0.333333, 0.266667, 0.114286])
            >>> X.l_stats(trim=(0, 1)).round(6)
            array([0.5     , 0.25    , 0.222222, 0.083333])

        Note:
            This should not be confused with the term *L-statistic*, which is
            sometimes used to describe any linear combination of order
            statistics.

        Args:
            *args:
                The shape parameter(s) for the distribution (see docstring
                of the instance object for more information)
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
            moments:
                The amount of L-moments to return. Defaults to 4.
            quad_opts:
                Optional dict of options to pass to
                [`scipy.integrate.quad`][scipy.integrate.quad].
            **kwds:
                Additional keyword arguments to pass to the distribution.


        See Also:
            - [`l_rv_generic.l_ratio`
            ][lmo.contrib.scipy_stats.l_rv_generic.l_ratio]
            - [`lmo.l_stats`][lmo.l_stats] - Unbiased sample estimation of
              L-stats.
        """
        r, s = l_stats_orders(moments)
        return self.l_ratio(
            r,
            s,
            *args,
            trim=trim,
            quad_opts=quad_opts,
            **kwds,
        )

    def l_loc(self, *args: Any, trim: lmt.ToTrim = 0, **kwds: Any) -> float:
        """
        L-location of the distribution, i.e. the 1st L-moment.

        Alias for `X.l_moment(1, ...)`.
        """
        if not any(clean_trim(trim)):
            return self.mean(*args, **kwds)

        return self.l_moment(1, *args, trim=trim, **kwds)

    def l_scale(self, *args: Any, trim: lmt.ToTrim = 0, **kwds: Any) -> float:
        """
        L-scale of the distribution, i.e. the 2nd L-moment.

        Alias for `X.l_moment(2, ...)`.
        """
        return self.l_moment(2, *args, trim=trim, **kwds)

    def l_skew(self, *args: Any, trim: lmt.ToTrim = 0, **kwds: Any) -> float:
        """L-skewness coefficient of the distribution; the 3rd L-moment ratio.

        Alias for `X.l_ratio(3, 2, ...)`.
        """
        return self.l_ratio(3, 2, *args, trim=trim, **kwds)

    def l_kurt(self, *args: Any, trim: lmt.ToTrim = 0, **kwds: Any) -> float:
        """L-kurtosis coefficient of the distribution; the 4th L-moment ratio.

        Alias for `X.l_ratio(4, 2, ...)`.
        """
        return self.l_ratio(4, 2, *args, trim=trim, **kwds)

    l_kurtosis = l_kurt

    def l_moments_cov(
        self,
        r_max: int,
        /,
        *args: Any,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
        **kwds: Any,
    ) -> _FloatND:
        r"""
        Variance/covariance matrix of the L-moment estimators.

        L-moments that are estimated from $n$ samples of a distribution with
        CDF $F$, converge to the multivariate normal distribution as the
        sample size $n \rightarrow \infty$.

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

        Here, $\vec{l}^{(s, t)} = \left[l^{(s,t)}_r, \dots,
        l^{(s,t)}_{r_{max}}\right]^T$ is a vector of estimated sample
        L-moments, and $\vec{\lambda}^{(s, t)}$ its theoretical ("true")
        counterpart.

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
            This function is not vectorized or parallelized.

            For small sample sizes ($n < 100$), the covariances of the
            higher-order L-moments ($r > 2$) can be biased. But this bias
            quickly disappears at roughly $n > 200$ (depending on the trim-
            and L-moment orders).

        Examples:
            >>> import lmo
            >>> from scipy.stats import distributions
            >>> X = distributions.expon()  # standard exponential distribution
            >>> X.l_moments_cov(4).round(6)
            array([[1.      , 0.5     , 0.166667, 0.083333],
                [0.5     , 0.333333, 0.166667, 0.083333],
                [0.166667, 0.166667, 0.133333, 0.083333],
                [0.083333, 0.083333, 0.083333, 0.071429]])

            >>> X.l_moments_cov(4, trim=(0, 1)).round(6)
            array([[0.333333, 0.125   , 0.      , 0.      ],
                [0.125   , 0.075   , 0.016667, 0.      ],
                [0.      , 0.016667, 0.016931, 0.00496 ],
                [0.      , 0.      , 0.00496 , 0.0062  ]])

        Args:
            r_max:
                The amount of L-moment orders to consider. If for example
                `r_max = 4`, the covariance matrix will be of shape `(4, 4)`,
                and the columns and rows correspond to the L-moments of order
                $r = 1, \dots, r_{max}$.
            *args:
                The shape parameter(s) for the distribution (see docstring
                of the instance object for more information)
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
                or floats.
            quad_opts:
                Optional dict of options to pass to
                [`scipy.integrate.quad`][scipy.integrate.quad].
            **kwds:
                Additional keyword arguments to pass to the distribution.

        Returns:
            cov: Covariance matrix, with shape `(r_max, r_max)`.

        Raises:
            RuntimeError: If the covariance matrix is invalid.

        References:
            - [J.R.M. Hosking (1990) - L-moments: Analysis and Estimation of
                Distributions Using Linear Combinations of Order Statistics
                ](https://jstor.org/stable/2345653)
            - [J.R.M. Hosking (2007) - Some theory and practical uses of
                trimmed L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
        """
        args, _, scale = self._parse_args(*args, **kwds)
        support = self._get_support(*args)
        cdf, _ = _get_xxf(self, *args)

        cov = l_moment_cov_from_cdf(
            cdf,
            r_max,
            trim=trim,
            support=support,
            quad_opts=quad_opts,
        )
        return scale**2 * cov

    def l_stats_cov(
        self,
        *args: Any,
        moments: int = 4,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
        **kwds: Any,
    ) -> _FloatND:
        r"""
        Similar to [`l_moments_cov`
        ][lmo.contrib.scipy_stats.l_rv_generic.l_moments_cov], but for the
        [`l_rv_generic.l_stats`][lmo.contrib.scipy_stats.l_rv_generic.l_stats].

        As the sample size $n \rightarrow \infty$, the L-moment ratio's are
        also distributed (multivariate) normally. The L-stats are defined to
        be L-moments for $r\le 2$, and L-ratio coefficients otherwise.

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

        Examples:
            Evaluate the LL-stats covariance matrix of the standard exponential
            distribution, for 0, 1, and 2 degrees of trimming.

            >>> import lmo
            >>> from scipy.stats import distributions
            >>> X = distributions.expon()  # standard exponential distribution
            >>> X.l_stats_cov().round(6)
            array([[1.      , 0.5     , 0.      , 0.      ],
                [0.5     , 0.333333, 0.111111, 0.055556],
                [0.      , 0.111111, 0.237037, 0.185185],
                [0.      , 0.055556, 0.185185, 0.21164 ]])
            >>> X.l_stats_cov(trim=(0, 1)).round(6)
            array([[ 0.333333,  0.125   , -0.111111, -0.041667],
                [ 0.125   ,  0.075   ,  0.      , -0.025   ],
                [-0.111111,  0.      ,  0.21164 ,  0.079365],
                [-0.041667, -0.025   ,  0.079365,  0.10754 ]])
            >>> X.l_stats_cov(trim=(0, 2)).round(6)
            array([[ 0.2     ,  0.066667, -0.114286, -0.02    ],
                [ 0.066667,  0.038095, -0.014286, -0.023333],
                [-0.114286, -0.014286,  0.228571,  0.04    ],
                [-0.02    , -0.023333,  0.04    ,  0.086545]])

            Note that with 0 trim the L-location is independent of the
            L-skewness and L-kurtosis. With 1 trim, the L-scale and L-skewness
            are independent. And with 2 trim, all L-stats depend on each other.

        Args:
            *args:
                The shape parameter(s) for the distribution (see docstring
                of the instance object for more information)
            moments:
                The amount of L-statistics to consider. Defaults to 4.
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
                or floats.
            quad_opts:
                Optional dict of options to pass to
                [`scipy.integrate.quad`][scipy.integrate.quad].
            **kwds:
                Additional keyword arguments to pass to the distribution.

        References:
            - [J.R.M. Hosking (1990) - L-moments: Analysis and Estimation of
                Distributions Using Linear Combinations of Order Statistics
                ](https://jstor.org/stable/2345653)
            - [J.R.M. Hosking (2007) - Some theory and practical uses of
                trimmed L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
        """
        args, _, scale = self._parse_args(*args, **kwds)
        support = self._get_support(*args)
        cdf, ppf = _get_xxf(self, *args)

        cov = l_stats_cov_from_cdf(
            cdf,
            moments,
            trim=trim,
            support=support,
            quad_opts=quad_opts,
            ppf=ppf,
        )
        return scale**2 * cov

    def l_moment_influence(
        self,
        r: lmt.ToOrder0D,
        /,
        *args: Any,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
        tol: float = 1e-8,
        **kwds: Any,
    ) -> _Fn1:
        r"""
        Returns the influence function (IF) of an L-moment.

        $$
        \psi_{\lambda^{(s, t)}_r | F}(x)
            = c^{(s,t)}_r
            \, F(x)^s
            \, \big( 1-{F}(x) \big)^t
            \, \tilde{P}^{(s,t)}_{r-1} \big( F(x) \big)
            \, x
            - \lambda^{(s,t)}_r
            \;,
        $$

        with $F$ the CDF, $\tilde{P}^{(s,t)}_{r-1}$ the shifted Jacobi
        polynomial, and

        $$
        c^{(s,t)}_r
            = \frac{r+s+t}{r} \frac{B(r, \, r+s+t)}{B(r+s, \, r+t)}
            \;,
        $$

        where $B$ is the (complete) Beta function.

        The proof is trivial, because population L-moments are
        [linear functionals](https://wikipedia.org/wiki/Linear_form).

        Notes:
            The order parameter `r` is not vectorized.

        Args:
            r:
                The L-moment order $r \in \mathbb{N}^+$..
            *args:
                The shape parameter(s) for the distribution (see docstring
                of the instance object for more information)
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
                or floats.
            quad_opts:
                Optional dict of options to pass to
                [`scipy.integrate.quad`][scipy.integrate.quad].
            tol:
                Values that are absolutely smaller than this will be rounded
                to zero.
            **kwds:
                Additional keyword arguments to pass to the distribution.

        Returns:
            influence_function:
                The (vectorized) influence function,
                $\psi_{\lambda^{(s, t)}_r | F}(x)$.

        See Also:
            - [`l_rv_generic.l_moment`
            ][lmo.contrib.scipy_stats.l_rv_generic.l_moment]
            - [`lmo.l_moment`][lmo.l_moment]

        References:
            - [Frank R. Hampel (1974) - The Influence Curve and its Role in
                Robust Estimation](https://doi.org/10.2307/2285666)

        """
        lm = self.l_moment(r, *args, trim=trim, quad_opts=quad_opts, **kwds)

        args, loc, scale = self._parse_args(*args, **kwds)
        cdf, _ = _get_xxf(self, *args, loc=loc, scale=scale)

        return l_moment_influence_from_cdf(
            cdf,
            r,
            trim=trim,
            support=self._get_support(*args),
            l_moment=lm,
            tol=tol,
        )

    def l_ratio_influence(
        self,
        r: lmt.ToOrder0D,
        k: lmt.ToOrder0D,
        /,
        *args: Any,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
        tol: float = 1e-8,
        **kwds: Any,
    ) -> _Fn1:
        r"""
        Returns the influence function (IF) of an L-moment ratio.

        $$
        \psi_{\tau^{(s, t)}_{r,k}|F}(x) = \frac{
            \psi_{\lambda^{(s, t)}_r|F}(x)
            - \tau^{(s, t)}_{r,k} \, \psi_{\lambda^{(s, t)}_k|F}(x)
        }{
            \lambda^{(s,t)}_k
        } \;,
        $$

        where the L-moment ratio is defined as

        $$
        \tau^{(s, t)}_{r,k} = \frac{
            \lambda^{(s, t)}_r
        }{
            \lambda^{(s, t)}_k
        } \;.
        $$

        Because IF's are a special case of the general Gâteuax derivative, the
        L-ratio IF is derived by applying the chain rule to the
        [L-moment IF][lmo.theoretical.l_moment_influence_from_cdf].


        Args:
            r:
                L-moment ratio order, i.e. the order of the numerator L-moment.
            k:
                Denominator L-moment order, defaults to 2.
            *args:
                The shape parameter(s) for the distribution (see docstring
                of the instance object for more information)
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
                or floats.
            quad_opts:
                Optional dict of options to pass to
                [`scipy.integrate.quad`][scipy.integrate.quad].
            tol:
                Values that are absolutely smaller than this will be rounded
                to zero.
            **kwds:
                Additional keyword arguments to pass to the distribution.

        Returns:
            influence_function:
                The influence function, with vectorized signature `() -> ()`.

        See Also:
            - [`l_rv_generic.l_ratio`
            ][lmo.contrib.scipy_stats.l_rv_generic.l_ratio]
            - [`lmo.l_ratio`][lmo.l_ratio]

        References:
            - [Frank R. Hampel (1974) - The Influence Curve and its Role in
                Robust Estimation](https://doi.org/10.2307/2285666)

        """
        rk = np.stack(np.broadcast_arrays(np.asarray(r), np.asarray(k)))
        lmr, lmk = self.l_moment(
            rk,
            *args,
            trim=trim,
            quad_opts=quad_opts,
            **kwds,
        )

        args, loc, scale = self._parse_args(*args, **kwds)
        cdf = _get_xxf(self, *args, loc=loc, scale=scale)[0]

        return l_ratio_influence_from_cdf(
            cdf,
            r,
            k,
            trim=trim,
            support=self._get_support(*args),
            l_moments=(lmr, lmk),
            quad_opts=quad_opts,
            tol=tol,
        )

    def _reduce_param_bounds(
        self,
        **kwds: onp.ToFloat,
    ) -> tuple[dict[str, onp.ToFloat], list[_Tuple2[float | None]]]:
        """
        Based on `scipy.stats.lmt.rv_continuous._reduce_func`.

        Convert fixed shape parameters to the standard numeric form: e.g. for
        stats.beta, shapes='a, b'. To fix `a`, the caller can give a value
        for `f0`, `fa` or 'fix_a'.  The following converts the latter two
        into the first (numeric) form.
        """
        kwds = kwds.copy()
        bounds: list[tuple[float | None, float | None]] = []

        for i, param in enumerate(self._param_info()):
            name = param.name
            if param.integrality:
                msg = f"integral parameter ({name!r}) fitting is not supported"
                raise NotImplementedError(msg)

            a, b = param.domain

            for key in (f"{i}", f"f{name}", f"fix_{name}"):
                if key in kwds:
                    if a == b:
                        msg = f"multiple fixed args given for {name!r}#{i}"
                        raise ValueError(msg)

                    val = float(kwds.pop(key))
                    if not (a <= val <= b):
                        msg = f"expected {a} <= {name} <= {b}, got {key}={val}"
                        raise ValueError(msg)

                    a = b = val

            bounds.append((
                None if np.isinf(a) else a,
                None if np.isinf(b) else b,
            ))

        return kwds, bounds

    def _l_gmm_error(
        self,
        theta: _FloatND,
        trim: _Tuple2[float],
        l_data: _FloatND,
        weights: _FloatND,
    ) -> float:
        """L-GMM objective function."""
        loc, scale, args = self._unpack_loc_scale(theta)
        if scale <= 0 or not self._argcheck(*args):
            return np.inf

        l_dist = self.l_moment(
            np.arange(1, len(weights) + 1),
            *args,
            loc=loc,
            scale=scale,
            trim=trim,
        )
        if np.any(np.isnan(l_dist)):
            msg = (
                f"Method of L-moments encountered a non-finite  {self.name}"
                f"L-moment and cannot continue."
            )
            raise ValueError(msg)

        err = l_data - l_dist
        return cast("float", err @ weights @ err)

    @overload
    def l_fit(
        self,
        data: onp.ToFloat1D,
        *args: float,
        n_extra: int = ...,
        trim: lmt.ToIntTrim = ...,
        full_output: Literal[True],
        fit_kwargs: Mapping[str, Any] | None = ...,
        **kwds: Any,
    ) -> tuple[float, ...]: ...
    @overload
    def l_fit(
        self,
        data: onp.ToFloat1D,
        *args: float,
        n_extra: int = ...,
        trim: lmt.ToIntTrim = ...,
        full_output: bool = ...,
        fit_kwargs: Mapping[str, Any] | None = ...,
        **kwds: Any,
    ) -> tuple[float, ...]: ...
    def l_fit(
        self,
        data: onp.ToFloat1D,
        *args: float,
        n_extra: int = 0,
        trim: lmt.ToIntTrim = 0,
        full_output: bool = False,
        fit_kwargs: Mapping[str, Any] | None = None,
        random_state: int | np.random.Generator | None = None,
        **kwds: Any,
    ) -> tuple[float, ...] | inference.GMMResult:
        """
        Return estimates of shape (if applicable), location, and scale
        parameters from data. The default estimation method is Method of
        L-moments (L-MM), but the Generalized Method of L-Moments
        (L-GMM) is also available (see the `n_extra` parameter).

        See ['lmo.inference.fit'][lmo.inference.fit] for details.

        Examples:
            Fitting standard normal samples Using scipy's default MLE
            (Maximum Likelihood Estimation) method:

            >>> import lmo
            >>> import numpy as np
            >>> from scipy.stats import norm
            >>> rng = np.random.default_rng(12345)
            >>> x = rng.standard_normal(200)
            >>> norm.fit(x)
            (0.0033254, 0.95554)

            Better results can be obtained different by using Lmo's L-MM
            (Method of L-moment):

            >>> norm.l_fit(x, random_state=rng)
            (0.0033145, 0.96179)
            >>> norm.l_fit(x, trim=1, random_state=rng)
            (0.0196385, 0.96861)

            To use more L-moments than the number of parameters, two in this
            case, `n_extra` can be used. This will use the L-GMM (Generalized
            Method of L-Moments), which results in slightly better estimates:

            >>> norm.l_fit(x, n_extra=1, random_state=rng)
            (0.0039747, .96233)
            >>> norm.l_fit(x, trim=1, n_extra=1, random_state=rng)
            (-0.00127874, 0.968547)

        Parameters:
            data:
                1-D array-like data to use in estimating the distribution
                parameters.
            *args:
                Starting value(s) for any shape-characterizing arguments (
                those not provided will be determined by a call to
                `fit(data)`).
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
            n_extra:
                The amount of extra L-moment conditions to use than the amount
                of parameters. If 0 (default), L-MM will be used.
                If >0, $k$-step L-GMM will be used.
            full_output:
                If set to True, a `LGMMResult` instance will
                be returned, instead of only a tuple with parameters.
            fit_kwargs:
                Additional keyword arguments to be passed to
                ['lmo.inference.fit'][lmo.inference.fit] or
                ['scipy.optimize.minimize'][scipy.optimize.minimize].
            random_state:
                Integer or [`numpy.random.Generator`][numpy.random.Generator]
                instance, used for Monte-Carlo simulation when `n_extra > 0`.
                If `None` (default), the `random_state` of this distribution
                will be used.
            **kwds:
                Special keyword arguments are recognized as holding certain
                parameters fixed:

                    - `f0...fn`: hold respective shape parameters fixed.
                    Alternatively, shape parameters to fix can be specified by
                    name. For example, if `self.shapes == "a, b"`, `fa` and
                    `fix_a` are equivalent to `f0`, and `fb` and `fix_b` are
                    equivalent to `f1`.
                    - `floc`: hold location parameter fixed to specified value.
                    - `fscale`: hold scale parameter fixed to specified value.

        Returns:
            result:
                Named tuple with estimates for any shape parameters (if
                applicable), followed by those for location and scale.
                For most random variables, shape statistics will be returned,
                but there are exceptions (e.g. ``norm``).
                If `full_output=True`, an instance of `LGMMResult` will be
                returned instead.

        See Also:
            - ['lmo.inference.fit'][lmo.inference.fit]

        References:
            - [Alvarez et al. (2023) - Inference in parametric models with
            many L-moments](https://doi.org/10.48550/arXiv.2210.04146)

        Todo:
            - Support integral parameters.

        """
        x = np.asarray_chkfinite(data)
        _, bounds = self._reduce_param_bounds(**kwds)

        if len(args) == len(bounds):
            args0 = args
        elif isinstance(self, lmt.rv_continuous):
            args0 = self.fit(x, *args, **kwds)
        else:
            # almost never works without custom (finite and tight) bounds...
            # ... and otherwise it'll runs for +-17 exa-eons
            from scipy.stats import fit as scipy_fit

            rv = cast("rv_discrete", self)
            bounds0 = [
                (-np.inf if a is None else a, np.inf if b is None else b)
                for a, b in bounds
            ]
            args0 = scipy_fit(rv, x, bounds=bounds0, guess=args or None).params

        r = np.arange(1, len(args0) + n_extra + 1)

        lmo_cache: dict[tuple[float, ...], _FloatND] = {}
        lmo_fn_ = _l_moment

        # temporary cache to speed up L-moment calculations with the same
        # shape args
        def lmo_fn(
            r_: onp.ArrayND[np.intp],
            *args_: float,
            **kwds: Unpack[_TrimKwargs],
        ) -> _FloatND:
            shapes, loc, scale = args_[:-2], args_[-2], args_[-1]

            # r and trim are constants, so it's safe to ignore them here
            if shapes in lmo_cache:
                lmbda_r = lmo_cache[shapes]
            else:
                lmbda_r = lmo_fn_(self, r_, *shapes, **kwds)
                lmbda_r.setflags(write=False)  # prevent cache corruption
                lmo_cache[shapes] = lmbda_r

            # ensure we're working with a copy
            lmbda_r = lmbda_r * scale  # noqa: PLR6104
            if loc != 0:
                lmbda_r[0] += loc
            return lmbda_r

        kwargs0: dict[str, Any] = {
            "bounds": bounds,
            "random_state": random_state or self.random_state,
        }
        if not len(self._shape_info()):
            # no shape params; weight matrix only depends linearly on scale
            # => weight matrix is constant between steps, use 1 step by default
            kwargs0["k"] = 1

        result = inference.fit(
            ppf=self.ppf,
            args0=args0,
            n_obs=x.size,
            l_moments=l_moment_est(x, r, trim=trim),
            r=r,
            trim=trim,
            l_moment_fn=lmo_fn,
            **(kwargs0 | dict(fit_kwargs or {})),
        )
        if full_output:
            return result

        return tuple(map(float, result.args))

    def l_fit_loc_scale(
        self,
        data: onp.ToFloatND,
        *args: Any,
        trim: lmt.ToTrim = 0,
        **kwds: Any,
    ) -> tuple[float, float]:
        """
        Estimate loc and scale parameters from data using the first two
        L-moments.

        Notes:
            The implementation mimics that of
            [`fit_loc_scale()`][scipy.stats.lmt.rv_continuous.fit_loc_scale]

        Args:
            data:
                Data to fit.
            *args:
                The shape parameter(s) for the distribution (see docstring of
                the instance object for more information).
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
            **kwds:
                Additional keyword arguments to pass to the distribution.

        Returns:
            loc_hat: Estimated location parameter for the data.
            scale_hat: Estimated scale parameter for the data.

        """
        l1, l2 = self.l_moment([1, 2], *args, trim=trim, **kwds)
        l1_hat, l2_hat = l_moment_est(data, [1, 2], trim=clean_trim(trim))

        scale_hat = l2_hat / l2
        with np.errstate(invalid="ignore"):
            loc_hat = l1_hat - scale_hat * l1

        if not np.isfinite(loc_hat):
            loc_hat = 0
        if not (np.isfinite(scale_hat) and scale_hat > 0):
            scale_hat = 1

        return loc_hat, scale_hat


class l_rv_frozen(PatchClass):  # noqa: D101
    dist: l_rv_generic
    args: tuple[Any, ...]
    kwds: Mapping[str, Any]

    @overload
    def l_moment(
        self,
        order: lmt.ToOrder0D,
        /,
        trim: lmt.ToTrim = ...,
        quad_opts: lmt.QuadOptions | None = ...,
    ) -> float: ...
    @overload
    def l_moment(
        self,
        order: lmt.ToOrderND,
        /,
        trim: lmt.ToTrim = ...,
        quad_opts: lmt.QuadOptions | None = ...,
    ) -> _FloatND: ...
    def l_moment(  # noqa: D102
        self,
        order: lmt.ToOrder,
        /,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
    ) -> float | _FloatND:
        return self.dist.l_moment(
            order,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            **self.kwds,
        )

    @overload
    def l_ratio(
        self,
        order: lmt.ToOrder0D,
        order_denom: lmt.ToOrder,
        /,
        trim: lmt.ToTrim = ...,
        quad_opts: lmt.QuadOptions | None = ...,
    ) -> float: ...
    @overload
    def l_ratio(
        self,
        order: lmt.ToOrderND,
        order_denom: lmt.ToOrder,
        /,
        trim: lmt.ToTrim = ...,
        quad_opts: lmt.QuadOptions | None = ...,
    ) -> _FloatND: ...
    def l_ratio(  # noqa: D102
        self,
        order: lmt.ToOrder,
        order_denom: lmt.ToOrder,
        /,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
    ) -> float | _FloatND:
        return self.dist.l_ratio(
            order,
            order_denom,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_stats(  # noqa: D102
        self,
        trim: lmt.ToTrim = 0,
        moments: int = 4,
        quad_opts: lmt.QuadOptions | None = None,
    ) -> np.float64 | _FloatND:
        return self.dist.l_stats(
            *self.args,
            trim=trim,
            moments=moments,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_loc(self, trim: lmt.ToTrim = 0) -> float:  # noqa: D102
        return self.dist.l_loc(*self.args, trim=trim, **self.kwds)

    def l_scale(self, trim: lmt.ToTrim = 0) -> float:  # noqa: D102
        return self.dist.l_scale(*self.args, trim=trim, **self.kwds)

    def l_skew(self, trim: lmt.ToTrim = 0) -> float:  # noqa: D102
        return self.dist.l_skew(*self.args, trim=trim, **self.kwds)

    def l_kurtosis(self, trim: lmt.ToTrim = 0) -> float:  # noqa: D102
        return self.dist.l_kurtosis(*self.args, trim=trim, **self.kwds)

    l_kurt = l_kurtosis

    def l_moments_cov(  # noqa: D102
        self,
        r_max: int,
        /,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
    ) -> _FloatND:
        return self.dist.l_moments_cov(
            r_max,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_stats_cov(  # noqa: D102
        self,
        moments: int = 4,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
    ) -> _FloatND:
        return self.dist.l_stats_cov(
            *self.args,
            moments=moments,
            trim=trim,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_moment_influence(  # noqa: D102
        self,
        r: lmt.ToOrder0D,
        /,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
        tol: float = 1e-8,
    ) -> _Fn1:
        return self.dist.l_moment_influence(
            r,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            tol=tol,
            **self.kwds,
        )

    def l_ratio_influence(  # noqa: D102
        self,
        r: lmt.ToOrder0D,
        k: lmt.ToOrder0D,
        /,
        trim: lmt.ToTrim = 0,
        quad_opts: lmt.QuadOptions | None = None,
        tol: float = 1e-8,
    ) -> _Fn1:
        return self.dist.l_ratio_influence(
            r,
            k,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            tol=tol,
            **self.kwds,
        )


def _l_moment(
    self: l_rv_generic,
    r: onp.ArrayND[np.intp],
    /,
    *args: Any,
    trim: _Tuple2[int] | _Tuple2[float] = (0, 0),
    quad_opts: lmt.QuadOptions | None = None,
) -> _FloatND:
    """
    Population L-moments of the standard distribution (i.e. `loc, scale = 0, 1`).

    Todo:
        Sparse caching: Use a priority queue with key `(self, args, r, trim)`.
        Prefer small `r` and small `sum(trim)`, skip fractional trim.
    """
    name = self.name
    if quad_opts is None and has_lm_func(name):
        with contextlib.suppress(NotImplementedError):
            return get_lm_func(name)(r, *trim, *args)

    cdf, ppf = _get_xxf(self, *args)

    if prefers_ppf(name):
        lmbda_r = l_moment_from_ppf(ppf, r, trim=trim, quad_opts=quad_opts)
    else:
        a, b = self._get_support(*args)  # pyright: ignore[reportPrivateUsage]
        with np.errstate(over="ignore", under="ignore"):
            lmbda_r = l_moment_from_cdf(
                cdf,
                r,
                trim=trim,
                support=(float(a), float(b)),
                ppf=ppf,
                quad_opts=quad_opts,
            )

    # re-wrap scalars in 0-d arrays (lmo.theoretical unpacks them)
    return np.asarray(lmbda_r)


def _get_xxf(
    self: l_rv_generic,
    /,
    *shape: float,
    loc: float = 0,
    scale: float = 1,
) -> _Tuple2[_Fn1]:
    assert scale > 0

    cdf_, ppf_ = self._cdf, self._ppf  # pyright: ignore[reportPrivateUsage]

    @overload
    def cdf(x: onp.ToFloat, /) -> float: ...
    @overload
    def cdf(x: onp.ToFloatND, /) -> _FloatND: ...
    def cdf(x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        z = (np.array([x]) - loc) / scale
        out = cdf_(z, *shape)
        return out.item() if out.ndim == 0 else out

    @overload
    def ppf(p: onp.ToFloat, /) -> float: ...
    @overload
    def ppf(p: onp.ToFloatND, /) -> _FloatND: ...
    def ppf(p: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        return ppf_(np.array([p]), *shape)[0] * scale + loc

    return cdf, ppf


def install() -> None:
    """
    Add the public methods from
    [`l_rv_generic`][`lmo.contrib.scipy_stats.l_rv_generic`] and
    [`l_lmt.rv_frozen`][`lmo.contrib.scipy_stats.l_lmt.rv_frozen`]
    to the `scipy.stats.rv_generic` and `scipy.stats.lmt.rv_frozen`
    types, respectively.
    """
    l_rv_generic.patch(cast("type[object]", lmt.rv_continuous.__base__))
    l_rv_frozen.patch(cast("type[object]", lmt.rv_frozen))
