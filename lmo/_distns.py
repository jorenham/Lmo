# pyright: reportIncompatibleMethodOverride=false

__all__ = ('l_rv_nonparametric', 'l_rv_generic', 'l_rv_frozen')

import functools
import math
import warnings
from collections.abc import Callable, Mapping
from typing import (
    Any,
    ClassVar,
    Final,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize  # type: ignore
from scipy.stats.distributions import (  # type: ignore
    rv_continuous,
    rv_frozen,
)

from ._poly import jacobi_series, roots
from ._utils import (
    broadstack,
    clean_order,
    clean_orders,
    clean_trim,
    l_stats_orders,
    moments_to_ratio,
)
from .diagnostic import l_ratio_bounds
from .theoretical import (
    l_moment_cov_from_cdf,
    l_moment_from_cdf,
    l_moment_influence_from_cdf,
    l_ratio_influence_from_cdf,
    l_stats_cov_from_cdf,
)
from .typing import (
    AnyInt,
    AnyTrim,
    FloatVector,
    IntVector,
    PolySeries,
    QuadOptions,
)

X = TypeVar('X', bound='l_rv_nonparametric')
F = TypeVar('F', bound=np.floating[Any])
M = TypeVar('M', bound=Callable[..., Any])
V = TypeVar('V', bound=float | npt.NDArray[np.float_])

_F_EPS: Final[np.float_] = np.finfo(float).eps


def _check_lmoments(l_r: npt.NDArray[np.floating[Any]], s: float, t: float):
    if (n := len(l_r)) < 2:
        msg = f'at least 2 L-moments required, got {n}'
        raise ValueError(msg)
    if n == 2:
        return

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


class l_rv_nonparametric(rv_continuous):  # noqa: N801
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

    def _weights(self, q: npt.ArrayLike) -> npt.NDArray[np.float_]:
        _q = np.asarray(q, np.float_)
        s, t = self._trim
        return np.where(
            (_q >= 0) & (_q <= 1),
            _q**s * (1 - _q) ** t,
            cast(float, getattr(self, 'badvalue', np.nan)),  # type: ignore
        )

    def _ppf(self, q: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return cast(npt.NDArray[np.float_], self._ppf_poly(q))

    def _isf(self, q: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return cast(npt.NDArray[np.float_], self._isf_poly(q))

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

            if np.ptp(q0) <= 1 / 4:
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
            A fitted [`l_rv_nonparametric`][lmo.l_rv_nonparametric] instance.

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


class PatchClass:
    patched: ClassVar[set[type[object]]] = set()

    @classmethod
    def patch(cls, base: type[object]) -> None:
        if not isinstance(base, type):
            msg = 'patch() argument must be a type'
            raise TypeError(msg)
        if base in cls.patched:
            msg = f'{base.__qualname__} already patched'
            raise TypeError(msg)

        for name, method in cls.__dict__.items():
            if name.startswith('__') or not callable(method):
                continue
            if hasattr(base, name):
                msg = f'{base.__qualname__}.{name}() already exists'
                raise TypeError(msg)
            setattr(base, name, method)

        cls.patched.add(base)


class l_rv_generic(PatchClass):  # noqa: N801
    """
    Additional methods that are patched into
    [`scipy.stats.rv_continuous`][scipy.stats.rv_continuous] and
    [`scipy.stats.rv_discrete`][scipy.stats.rv_discrete].
    """

    _cdf: Callable[..., float]
    _ppf: Callable[..., float]
    _argcheck: Callable[..., int]
    _fitstart: Callable[..., tuple[float, ...]]
    _get_support: Callable[..., tuple[float, float]]
    _parse_args: Callable[..., tuple[tuple[Any, ...], float, float]]
    _unpack_loc_scale: Callable[
        [npt.ArrayLike],
        tuple[float, float, tuple[float, ...]],
    ]
    fit: Callable[..., tuple[float, ...]]
    mean: Callable[..., float]
    numargs: int

    def _get_xdf(self, *args: Any, loc: float = 0, scale: float = 1) -> tuple[
        Callable[[float], float],
        Callable[[float], float],
    ]:
        assert scale > 0

        _cdf, _ppf = self._cdf, self._ppf
        if args or loc != 0 or scale != 1:
            def cdf(x: float, /) -> float:
                return _cdf((x - loc) / scale, *args)

            def ppf(q: float, /):
                return _ppf(q, *args) * scale + loc
        else:
            cdf, ppf = _cdf, _ppf

        return cdf, ppf

    @overload
    def l_moment(
        self,
        order: AnyInt,
        /,
        *args: Any,
        trim: AnyTrim = ...,
        quad_opts: QuadOptions | None = ...,
        **kwds: Any,
    ) -> np.float_: ...

    @overload
    def l_moment(
        self,
        order: IntVector,
        /,
        *args: Any,
        trim: AnyTrim = ...,
        quad_opts: QuadOptions | None = ...,
        **kwds: Any,
    ) -> npt.NDArray[np.float_]: ...

    def l_moment(
        self,
        r: AnyInt | IntVector,
        /,
        *args: Any,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        **kwds: Any,
    ) -> np.float_ | npt.NDArray[np.float_]:
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
            15.000000...

            Discrete distributions are also supported, e.g. the Binomial
            distribution:

            >>> from scipy.stats import binom
            >>> binom(10, .6).l_moment([1, 2, 3, 4]).round(6)
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
        rs = clean_orders(np.asanyarray(r))

        args, loc, scale = self._parse_args(*args, **kwds)
        support = self._get_support(*args)
        cdf, ppf = self._get_xdf(*args)

        lmda = np.asarray(l_moment_from_cdf(
            cdf,
            rs,
            trim=trim,
            support=support,
            ppf=ppf,
            quad_opts=quad_opts,
        ))
        lmda[rs == 1] += loc
        lmda[rs > 1] *= scale
        return lmda[()]  # convert back to scalar if needed

    @overload
    def l_ratio(
        self,
        order: AnyInt,
        order_denom: AnyInt | IntVector,
        /,
        *args: Any,
        trim: AnyTrim = ...,
        quad_opts: QuadOptions | None = ...,
        **kwds: Any,
    ) -> np.float_: ...

    @overload
    def l_ratio(
        self,
        order: IntVector,
        order_denom: AnyInt | IntVector,
        /,
        *args: Any,
        trim: AnyTrim = ...,
        quad_opts: QuadOptions | None = ...,
        **kwds: Any,
    ) -> npt.NDArray[np.float_]: ...


    def l_ratio(
        self,
        r: AnyInt | IntVector,
        k: AnyInt | IntVector,
        /,
        *args: Any,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        **kwds: Any,
    ) -> np.float_ | npt.NDArray[np.float_]:
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
            0.5227232...
            >>> X.l_ratio(2, 1)
            0.2928932...
            >>> X.l_ratio(2, 1, trim=(0, 1))
            0.2752551...

            And similarly, for the (discrete) Poisson distribution with rate
            parameter set to 2, the L-CF and LL-CV evaluate to:

            >>> X = distributions.poisson(2)
            >>> X.std() / X.mean()
            0.7071067...
            >>> X.l_ratio(2, 1)
            0.3857527...
            >>> X.l_ratio(2, 1, trim=(0, 1))
            0.4097538...

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
            - [`lmo.l_rv_generic.l_moment`][lmo.l_rv_generic.l_moment]
            - [`lmo.l_ratio`][lmo.l_ratio] - Sample L-moment ratio estimator
        """
        rs = broadstack(r, k)
        lms = self.l_moment(
            rs,
            *args,
            trim=trim,
            quad_opts=quad_opts,
            **kwds,
        )
        return moments_to_ratio(rs, lms)

    def l_stats(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        moments: int = 4,
        quad_opts: QuadOptions | None = None,
        **kwds: Any,
    ) -> npt.NDArray[np.float_]:
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
            >>> X.l_stats(trim=(0, 1/2)).round(6)
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
            - [`lmo.l_rv_generic.l_ratio`][lmo.l_rv_generic.l_ratio]
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

    def l_loc(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        **kwds: Any,
    ) -> float:
        """
        L-location of the distribution, i.e. the 1st L-moment.

        Alias for `X.l_moment(1, ...)`.
        """
        if not any(clean_trim(trim)):
            return self.mean(*args, **kwds)

        return float(self.l_moment(1, *args, trim=trim, **kwds))

    def l_scale(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        **kwds: Any,
    ) -> float:
        """
        L-scale of the distribution, i.e. the 2nd L-moment.

        Alias for `X.l_moment(2, ...)`.
        """
        return float(self.l_moment(2, *args, trim=trim, **kwds))

    def l_skew(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        **kwds: Any,
    ) -> float:
        """L-skewness coefficient of the distribution; the 3rd L-moment ratio.

        Alias for `X.l_ratio(3, 2, ...)`.
        """
        return float(self.l_ratio(3, 2, *args, trim=trim, **kwds))

    def l_kurtosis(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        **kwds: Any,
    ) -> float:
        """L-kurtosis coefficient of the distribution; the 4th L-moment ratio.

        Alias for `X.l_ratio(4, 2, ...)`.
        """
        return float(self.l_ratio(4, 2, *args, trim=trim, **kwds))

    def l_moments_cov(
        self,
        r_max: int,
        /,
        *args: Any,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        **kwds: Any,
    ) -> npt.NDArray[np.float_]:
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
        args, _, scale = cast(
            tuple[tuple[float, ...], float, float],
            self._parse_args(*args, **kwds),
        )
        support = self._get_support(*args)
        cdf, _ = self._get_xdf(*args)

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
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        **kwds: Any,
    ) -> npt.NDArray[np.float_]:
        r"""
        Similar to [`l_moments_cov`][lmo.l_rv_generic.l_moments_cov], but
        for the [`l_stats`][lmo.l_rv_generic.l_stats].

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
        cdf, ppf = self._get_xdf(*args)

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
        r: AnyInt,
        /,
        *args: Any,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        tol: float = 1e-8,
        **kwds: Any,
    ) -> Callable[[V], V]:
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
            - [`lmo.l_rv_generic.l_moment`][lmo.l_rv_generic.l_moment]
            - [`lmo.l_moment`][lmo.l_moment]

        References:
            - [Frank R. Hampel (1974) - The Influence Curve and its Role in
                Robust Estimation](https://doi.org/10.2307/2285666)

        """
        lm = self.l_moment(r, *args, trim=trim, quad_opts=quad_opts, **kwds)

        args, loc, scale = self._parse_args(*args, **kwds)
        cdf = cast(
            Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
            self._get_xdf(*args, loc=loc, scale=scale)[0],
        )

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
        r: AnyInt,
        k: AnyInt,
        /,
        *args: Any,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        tol: float = 1e-8,
        **kwds: Any,
    ) -> Callable[[V], V]:
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
            - [`lmo.l_rv_generic.l_ratio`][lmo.l_rv_generic.l_ratio]
            - [`lmo.l_ratio`][lmo.l_ratio]

        References:
            - [Frank R. Hampel (1974) - The Influence Curve and its Role in
                Robust Estimation](https://doi.org/10.2307/2285666)

        """
        lmr, lmk = self.l_moment(
            [r, k],
            *args,
            trim=trim,
            quad_opts=quad_opts,
            **kwds,
        )

        args, loc, scale = self._parse_args(*args, **kwds)
        cdf = cast(
            Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
            self._get_xdf(*args, loc=loc, scale=scale)[0],
        )

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

    def _l_moment_error(
        self,
        theta: npt.NDArray[np.float_],
        trim: tuple[float, float],
        l_data: npt.NDArray[np.float_],
    ) -> float:
        loc, scale, args = self._unpack_loc_scale(theta)
        if scale <= 0 or not self._argcheck(*args):
            return np.inf

        l_dist = self.l_moment(
            np.arange(1, len(l_data) + 1),
            *args,
            loc=loc,
            scale=scale,
            trim=trim,
        )
        if np.any(np.isnan(l_dist)):
            msg = (
                'Method of L-moments encountered a non-finite distribution '
                'L-moment and cannot continue.'
            )
            raise ValueError(msg)


        return (
            ((l_data - l_dist) / np.maximum(np.abs(l_data), 1e-8))**2
        ).sum()


    def fit_l(
        self,
        data: npt.ArrayLike,
        *args: float,
        trim: AnyTrim = (0, 0),
        optimizer: str | Callable[..., OptimizeResult] = 'Nelder-Mead',
    ) -> tuple[float, ...]:
        """
        Return estimates of shape (if applicable), location, and scale
        parameters from data, using Method of L-Moments (LMM).

        Notes:
            The implementation mimics that of
            [`fit(method='MM')`][scipy.stats.rv_continuous.fit]

        Examples:
            Fitting of the generalized extreme value (GEV) distribution with
            MLE (maximum likelihood estimation, which is used by default in
            `.fit`) vs LMM with different degrees of trimming.

            >>> from scipy.stats import genextreme
            >>> c, loc, scale = -0.5, 0, 1
            >>> data = genextreme(c, loc, scale).rvs(1000, random_state=12345)
            >>> genextreme.fit(data)
            (-0.4670..., 0.0294..., 1.0241...)
            >>> genextreme.fit_l(data)
            (-0.447..., 0.085..., 1.036...)
            >>> genextreme.fit_l(data, trim=(0, 1))
            (-0.483..., 0.022..., 1.017...)
            >>> genextreme.fit_l(data, trim=(0, 2))
            (-0.486..., 0.016..., 1.016...)

        Args:
            data:
                Data to fit.
            *args:
                The shape parameter(s) for the distribution (see docstring of
                the instance object for more information).
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.
            optimizer:
                See the `method` parameter in
                [`scipy.optimize.minimize`][scipy.optimize.minimize]. Defaults
                to `'Nelder-Mead'`. Note that other methods (e.g.`'BFGS'`
                or `'SLSQP'`) can be a lot faster, but may not be as accurate.

        Returns:
            Tuple of floats with estimates of the shape parameters (if
            applicable), followed by those for location and scale.

        """
        _x = np.sort(data, kind='stable')
        _trim = clean_trim(trim)

        from ._lm import l_moment

        l_data = l_moment(_x, np.arange(self.numargs + 2) + 1, trim=_trim)

        theta0 = args or self._fitstart(_x)[:-2]
        theta0 = theta0 + self.fit_l_loc_scale(_x, *theta0, trim=_trim)

        theta = cast(
            npt.NDArray[np.float_],
            minimize(
                self._l_moment_error,
                theta0,
                (_trim, l_data),
                method=optimizer,
            ).x,  # type: ignore
        )

        return tuple(theta)

    def fit_l_loc_scale(
        self,
        data: npt.ArrayLike,
        *args: Any,
        trim: AnyTrim = (0, 0),
    ) -> tuple[float, float]:
        """
        Estimate loc and scale parameters from data using 1st and 2nd
        L-moments.

        Notes:
            The implementation mimics that of
            [`fit_loc_scale()`][scipy.stats.rv_continuous.fit_loc_scale]

        Args:
            data:
                Data to fit.
            *args:
                The shape parameter(s) for the distribution (see docstring of
                the instance object for more information).
            trim:
                Left- and right- trim. Can be scalar or 2-tuple of
                non-negative int or float.

        Returns:
            loc_hat: Estimated location parameter for the data.
            scale_hat: Estimated scale parameter for the data.

        """
        l1, l2 = self.l_moment([1, 2], *args, trim=trim)

        from ._lm import l_moment
        l1_hat, l2_hat = l_moment(data, [1, 2], trim=clean_trim(trim))

        scale_hat = l2_hat / l2
        with np.errstate(invalid='ignore'):
            loc_hat = l1_hat - scale_hat * l1

        if not np.isfinite(loc_hat):
            loc_hat = 0
        if not (np.isfinite(scale_hat) and scale_hat > 0):
            scale_hat = 1

        return loc_hat, scale_hat


class l_rv_frozen(PatchClass):  # noqa: N801
    dist: l_rv_generic
    args: tuple[Any, ...]
    kwds: Mapping[str, Any]

    @overload
    def l_moment(
        self,
        order: AnyInt,
        /,
        trim: AnyTrim = ...,
        quad_opts: QuadOptions | None = ...,
    ) -> np.float_: ...

    @overload
    def l_moment(
        self,
        order: IntVector,
        /,
        trim: AnyTrim = ...,
        quad_opts: QuadOptions | None = ...,
    ) -> npt.NDArray[np.float_]: ...

    def l_moment(
        self,
        order: AnyInt | IntVector,
        /,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
    ) -> np.float_ | npt.NDArray[np.float_]:
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
        order: AnyInt,
        order_denom: AnyInt | IntVector,
        /,
        trim: AnyTrim = ...,
        quad_opts: QuadOptions | None = ...,
    ) -> np.float_: ...

    @overload
    def l_ratio(
        self,
        order: IntVector,
        order_denom: AnyInt | IntVector,
        /,
        trim: AnyTrim = ...,
        quad_opts: QuadOptions | None = ...,
    ) -> npt.NDArray[np.float_]: ...

    def l_ratio(
        self,
        order: AnyInt | IntVector,
        order_denom: AnyInt | IntVector,
        /,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
    ) -> np.float_ | npt.NDArray[np.float_]:
        return self.dist.l_ratio(
            order,
            order_denom,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_stats(
        self,
        trim: AnyTrim = (0, 0),
        moments: int = 4,
        quad_opts: QuadOptions | None = None,
    ) -> np.float_ | npt.NDArray[np.float_]:
        return self.dist.l_stats(
            *self.args,
            trim=trim,
            moments=moments,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_loc(self, trim: AnyTrim = (0, 0)) -> float:
        return self.dist.l_loc(*self.args, trim=trim, **self.kwds)


    def l_scale(self, trim: AnyTrim = (0, 0)) -> float:
        return self.dist.l_scale(*self.args, trim=trim, **self.kwds)


    def l_skew(self, trim: AnyTrim = (0, 0)) -> float:
        return self.dist.l_skew(*self.args, trim=trim, **self.kwds)


    def l_kurtosis(self, trim: AnyTrim = (0, 0)) -> float:
        return self.dist.l_kurtosis(*self.args, trim=trim, **self.kwds)

    def l_moments_cov(
        self,
        r_max: int,
        /,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
    ) -> npt.NDArray[np.float_]:
        return self.dist.l_moments_cov(
            r_max,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_stats_cov(
        self,
        moments: int = 4,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
    ) -> npt.NDArray[np.float_]:
        return self.dist.l_stats_cov(
            *self.args,
            moments=moments,
            trim=trim,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_moment_influence(
        self,
        r: AnyInt,
        /,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        tol: float = 1e-8,
    ) -> Callable[[V], V]:
        return self.dist.l_moment_influence(
            r,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            tol=tol,
            **self.kwds,
        )

    def l_ratio_influence(
        self,
        r: AnyInt,
        k: AnyInt,
        /,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        tol: float = 1e-8,
    ) -> Callable[[V], V]:
        return self.dist.l_ratio_influence(
            r,
            k,
            *self.args,
            trim=trim,
            quad_opts=quad_opts,
            tol=tol,
            **self.kwds,
        )

l_rv_generic.patch(cast(type[object], rv_continuous.__base__))
l_rv_frozen.patch(cast(type[object], rv_frozen))
