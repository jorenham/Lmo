# pyright: reportIncompatibleMethodOverride=false

__all__ = ('l_rv_nonparametric', 'l_rv_generic', 'l_rv_frozen')

import functools
import math
import warnings
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Final, SupportsIndex, TypeVar, cast, overload

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt
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
from .theoretical import l_moment_from_cdf
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
            if name.startswith('_') or not callable(method):
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

    _get_support: Callable[..., tuple[float, float]]
    _cdf: Callable[..., float]
    _ppf: Callable[..., float]
    mean: Callable[..., float]

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

        args, loc, scale = cast(
            tuple[tuple[float, ...], float, float],
            self._parse_args(*args, **kwds),  # type: ignore
        )
        support = self._get_support(*args)

        _cdf, _ppf = self._cdf, self._ppf
        if args:
            def cdf(x: float, /) -> float:
                return _cdf(x, *args)

            def ppf(q: float, /):
                return _ppf(q, *args)
        else:
            cdf, ppf = _cdf, _ppf

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
        lms = cast(
            npt.NDArray[np.float_],
            self.l_moment(  # type: ignore
                rs,
                *args,
                trim=trim,
                quad_opts=quad_opts,
                **kwds,
            ),
        )
        return moments_to_ratio(rs, lms)

    def l_stats(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        moments: int = 4,
        quad_opts: QuadOptions | None = None,
        **kwds: Any,
    ) -> np.float_ | npt.NDArray[np.float_]:
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
        return cast(
            npt.NDArray[np.float_],
            self.l_ratio(  # type: ignore
                r,
                s,
                *args,
                trim=trim,
                quad_opts=quad_opts,
                **kwds,
            ),
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


l_rv_generic.patch(cast(type[object], rv_continuous.__base__))
l_rv_frozen.patch(cast(type[object], rv_frozen))
