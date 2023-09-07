# pyright: reportIncompatibleMethodOverride=false

__all__ = ('l_sample_rv', 'rv_generic_extra', 'rv_frozen_extra')

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

X = TypeVar('X', bound='l_sample_rv')
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


class l_sample_rv(rv_continuous):  # noqa: N801
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
            msg = 'invalid l_sample_rv: ppf(0) >= ppf(1)'
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
    ) -> 'l_sample_rv':
        r"""
        Estimate L-moment from the samples, and return a new `l_sample_rv`
        instance.

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
            A fitted [`l_sample_rv`][lmo.l_sample_rv] instance.

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


class rv_generic_extra(PatchClass):  # noqa: N801
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
        order: AnyInt | IntVector,
        /,
        *args: Any,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        **kwds: Any,
    ) -> np.float_ | npt.NDArray[np.float_]:
        """L-moment(s) of distribution of specified order(s).

        Parameters
        ----------
        order : array_like
            Order(s) of L-moment(s).
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : float, optional
            location parameter (default=0)
        scale : float, optional
            scale parameter (default=1)
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        lm : ndarray or scalar
            The calculated L-moment(s).

        """  # noqa: D416
        rs = clean_orders(np.asanyarray(order))

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

        lm = np.asarray(l_moment_from_cdf(
            cdf,
            rs,
            trim=trim,
            support=support,
            ppf=ppf,
            quad_opts=quad_opts,
        ))
        lm[rs == 1] += loc
        lm[rs > 1] *= scale
        return lm[()]  # convert back to scalar if needed

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
        order: AnyInt | IntVector,
        order_denom: AnyInt | IntVector,
        /,
        *args: Any,
        trim: AnyTrim = (0, 0),
        quad_opts: QuadOptions | None = None,
        **kwds: Any,
    ) -> np.float_ | npt.NDArray[np.float_]:
        """L-moment ratio('s) of distribution of specified order(s).

        Parameters
        ----------
        order : array_like
            Order(s) of L-moment(s).
        order_denom : array_like
            Order(s) of L-moment denominator(s).
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : float, optional
            location parameter (default=0)
        scale : float, optional
            scale parameter (default=1)
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        tm : ndarray or scalar
            The calculated L-moment ratio('s).

        """  # noqa: D416
        rs = broadstack(order, order_denom)
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
        """L-moments (order <= 2) and L-moment ratio's (order > 2).

        By default, the first `num = 4` L-stats are calculated. This is
        equivalent to `l_ratio([1, 2, 3, 4], [0, 0, 2, 2], *, **)`, i.e. the
        L-location, L-scale, L-skew, and L-kurtosis.

        Parameters
        ----------
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : float, optional
            location parameter (default=0)
        scale : float, optional
            scale parameter (default=1)
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))
        moments : int, optional
            the amount of L-moment stats to compute (default=4)

        Returns
        -------
        tm : ndarray or scalar
            The calculated L-moment ratio('s).

        """  # noqa: D416
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
        """L-location of the distribution, i.e. the 1st L-moment.

        Without trim (default), the L-location is equivalent to the mean.

        Parameters
        ----------
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : float, optional
            location parameter (default=0)
        scale : float, optional
            scale parameter (default=1)
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        l_loc : float
            The L-location of the distribution.

        """  # noqa: D416
        if not any(clean_trim(trim)):
            return self.mean(*args, **kwds)

        return float(self.l_moment(1, *args, trim=trim, **kwds))

    def l_scale(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        **kwds: Any,
    ) -> float:
        """L-scale of the distribution, i.e. the 2nd L-moment.

        Without trim (default), the L-location is equivalent to half the Gini
        mean (absolute) difference (GMD).

        Just like the standard deviation, the L-scale is location-invariant,
        and varies proportionally to positive scaling.

        Parameters
        ----------
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : float, optional
            location parameter (default=0)
        scale : float, optional
            scale parameter (default=1)
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        l_scale : float
                The L-scale of the distribution.

        """  # noqa: D416
        return float(self.l_moment(2, *args, trim=trim, **kwds))

    def l_skew(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        **kwds: Any,
    ) -> float:
        """L-skewness coefficient of the distribution; the 3rd L-moment ratio.

        Parameters
        ----------
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : float, optional
            location parameter (default=0)
        scale : float, optional
            scale parameter (default=1)
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        l_skew : float
            The L-skewness coefficient of the distribution.

        """  # noqa: D416
        return float(self.l_ratio(3, 2, *args, trim=trim, **kwds))

    def l_kurtosis(
        self,
        *args: Any,
        trim: AnyTrim = (0, 0),
        **kwds: Any,
    ) -> float:
        """L-kurtosis coefficient of the distribution; the 4th L-moment ratio.

        Parameters
        ----------
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : float, optional
            location parameter (default=0)
        scale : float, optional
            scale parameter (default=1)
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        l_kurtosis : float
            The L-kurtosis coefficient of the distribution.

        """  # noqa: D416
        return float(self.l_ratio(4, 2, *args, trim=trim, **kwds))



class rv_frozen_extra(PatchClass):  # noqa: N801
    dist: rv_generic_extra
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
        """L-moment(s) of distribution of specified order(s).

        Parameters
        ----------
        order : array_like
            Order(s) of L-moment(s).
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        lm : ndarray or scalar
            The calculated L-moment(s).

        """  # noqa: D416
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
        """L-moment ratio('s) of distribution of specified order(s).

        Parameters
        ----------
        order : array_like
            Order(s) of L-moment(s).
        order_denom : array_like
            Order(s) of L-moment denominator(s).
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        tm : ndarray or scalar
            The calculated L-moment ratio('s).

        """  # noqa: D416
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
        """L-moments (order <= 2) and L-moment ratio's (order > 2).

        By default, the first `num = 4` L-stats are calculated. This is
        equivalent to `l_ratio([1, 2, 3, 4], [0, 0, 2, 2], *, **)`, i.e. the
        L-location, L-scale, L-skew, and L-kurtosis.

        Parameters
        ----------
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))
        moments : int, optional
            the amount of L-moment stats to compute (default=4)

        Returns
        -------
        tm : ndarray or scalar
            The calculated L-moment ratio('s).

        """  # noqa: D416
        return self.dist.l_stats(
            *self.args,
            trim=trim,
            moments=moments,
            quad_opts=quad_opts,
            **self.kwds,
        )

    def l_loc(self, trim: AnyTrim = (0, 0)) -> float:
        """L-location of the distribution, i.e. the 1st L-moment.

        Without trim (default), the L-location is equivalent to the mean.

        Parameters
        ----------
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        l_loc : float
            The L-location of the distribution.

        """  # noqa: D416
        return self.dist.l_loc(*self.args, trim=trim, **self.kwds)


    def l_scale(self, trim: AnyTrim = (0, 0)) -> float:
        """L-scale of the distribution, i.e. the 2nd L-moment.

        Without trim (default), the L-location is equivalent to half the Gini
        mean (absolute) difference (GMD).

        Just like the standard deviation, the L-scale is location-invariant,
        and varies proportionally to positive scaling.

        Parameters
        ----------
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        l_scale : float
            The L-scale of the distribution.

        """  # noqa: D416
        return self.dist.l_scale(*self.args, trim=trim, **self.kwds)


    def l_skew(self, trim: AnyTrim = (0, 0)) -> float:
        """L-skewness coefficient of the distribution; the 3rd L-moment ratio.

        Parameters
        ----------
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        l_skew : float
            The L-skewness coefficient of the distribution.

        """  # noqa: D416
        return self.dist.l_skew(*self.args, trim=trim, **self.kwds)


    def l_kurtosis(self, trim: AnyTrim = (0, 0)) -> float:
        """L-kurtosis coefficient of the distribution; the 4th L-moment ratio.

        Parameters
        ----------
        trim : float or tuple, optional
            left- and right- trim (default=(0, 0))

        Returns
        -------
        l_kurtosis : float
            The L-kurtosis coefficient of the distribution.

        """  # noqa: D416
        return self.dist.l_kurtosis(*self.args, trim=trim, **self.kwds)


rv_generic_extra.patch(cast(type[object], rv_continuous.__base__))
rv_frozen_extra.patch(cast(type[object], rv_frozen))
