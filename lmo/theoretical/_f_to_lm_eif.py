from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt

import lmo.typing.np as lnpt
import lmo.typing.scipy as lspt
from lmo._poly import eval_sh_jacobi
from lmo._utils import clean_order, clean_trim, round0
from ._f_to_lm import l_moment_from_cdf
from ._utils import ALPHA, l_const, tighten_cdf_support


if TYPE_CHECKING:
    import lmo.typing as lmt


__all__ = ['l_moment_influence_from_cdf', 'l_ratio_influence_from_cdf']


_T = TypeVar('_T')
_T_x = TypeVar('_T_x', bound=float | npt.NDArray[np.float64])

_Pair: TypeAlias = tuple[_T, _T]
_Fn1: TypeAlias = Callable[[float], float | lnpt.Float]
_ArrF8: TypeAlias = npt.NDArray[np.float64]


def l_moment_influence_from_cdf(
    cdf: Callable[[_ArrF8], _ArrF8],
    r: lmt.AnyOrder,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    l_moment: float | np.float64 | None = None,
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
    tol: float = 1e-8,
) -> Callable[[_T_x], _T_x]:
    r"""
    Influence Function (IF) of a theoretical L-moment.

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

    with $F$ the CDF, $\tilde{P}^{(s,t)}_{r-1}$ the shifted Jacobi polynomial,
    and

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
        cdf: Vectorized cumulative distribution function (CDF).
        r: The L-moment order. Must be a non-negative integer.
        trim: Left- and right- trim lengths. Defaults to (0, 0).

    Other parameters:
        support:
            The subinterval of the nonzero domain of `cdf`.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        l_moment:
            The relevant L-moment to use. If not provided, it is calculated
            from the CDF.
        alpha: Two-sided quantile to split the integral at.
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The influence function, with vectorized signature `() -> ()`.

    See Also:
        - [`l_moment_from_cdf`][lmo.theoretical.l_moment_from_cdf]
        - [`lmo.l_moment`][lmo.l_moment]

    """
    _r = clean_order(int(r))
    if _r == 0:

        def influence0(x: _T_x, /) -> _T_x:
            """
            L-moment Influence Function for `r=0`.

            Args:
                x: Scalar or array-like of sample observarions.

            Returns:
                out
            """
            _x = np.asanyarray(x, np.float64)[()]
            return cast(_T_x, _x * 0.0 + 0.0)  # :+)

        return influence0

    s, t = clean_trim(trim)

    if l_moment is None:
        lm = l_moment_from_cdf(
            cast(Callable[[float], float], cdf),
            _r,
            trim=(s, t),
            support=support,
            quad_opts=quad_opts,
            alpha=alpha,
        )
    else:
        lm = l_moment

    a, b = support or tighten_cdf_support(cast(_Fn1, cdf), support)
    c = l_const(_r, s, t)

    def influence(x: _T_x, /) -> _T_x:
        _x = np.asanyarray(x, np.float64)
        q = np.piecewise(
            _x,
            [_x <= a, (_x > a) & (_x < b), _x >= b],
            [0, cdf, 1],
        )
        w = round0(c * q**s * (1 - q) ** t, tol)

        # cheat a bit and replace 0 * inf by 0, ensuring convergence if s or t
        alpha = w * eval_sh_jacobi(_r - 1, t, s, q) * np.where(w, _x, 0)

        return cast(_T_x, round0(alpha - lm, tol)[()])

    influence.__doc__ = (
        f'Theoretical influence function for L-moment with {r=} and {trim=}.'
    )

    return influence


def l_ratio_influence_from_cdf(
    cdf: Callable[[_ArrF8], _ArrF8],
    r: lmt.AnyOrder,
    k: lmt.AnyOrder = 2,
    /,
    trim: lmt.AnyTrim = 0,
    *,
    support: _Pair[float] | None = None,
    l_moments: _Pair[float] | None = None,
    quad_opts: lspt.QuadOptions | None = None,
    alpha: float = ALPHA,
    tol: float = 1e-8,
) -> Callable[[_T_x], _T_x]:
    r"""
    Construct the influence function of a theoretical L-moment ratio.

    $$
    \psi_{\tau^{(s, t)}_{r,k}|F}(x) = \frac{
        \psi_{\lambda^{(s, t)}_r|F}(x)
        - \tau^{(s, t)}_{r,k} \, \psi_{\lambda^{(s, t)}_k|F}(x)
    }{
        \lambda^{(s,t)}_k
    } \;,
    $$

    where the generalized L-moment ratio is defined as

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
        cdf: Vectorized cumulative distribution function (CDF).
        r: L-moment ratio order, i.e. the order of the numerator L-moment.
        k: Denominator L-moment order, defaults to 2.
        trim: Left- and right- trim lengths. Defaults to (0, 0).

    Other parameters:
        support:
            The subinterval of the nonzero domain of `cdf`.
        l_moments:
            The L-moments corresponding to $r$ and $k$. If not provided, they
            are calculated from the CDF.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].
        alpha: Two-sided quantile to split the integral at.
        tol: Zero-roundoff absolute threshold.

    Returns:
        influence_function:
            The influence function, with vectorized signature `() -> ()`.

    See Also:
        - [`l_ratio_from_cdf`][lmo.theoretical.l_ratio_from_cdf]
        - [`lmo.l_ratio`][lmo.l_ratio]

    """
    _r, _k = clean_order(int(r)), clean_order(int(k))

    kwds: dict[str, Any] = {'support': support, 'quad_opts': quad_opts}

    if l_moments is None:
        l_r, l_k = l_moment_from_cdf(
            cast(Callable[[float], float], cdf),
            [_r, _k],
            trim=trim,
            alpha=alpha,
            **kwds,
        )
    else:
        l_r, l_k = l_moments

    if_r = l_moment_influence_from_cdf(
        cdf,
        _r,
        trim,
        l_moment=l_r,
        tol=0,
        **kwds,
    )
    if_k = l_moment_influence_from_cdf(
        cdf,
        _k,
        trim,
        l_moment=l_k,
        tol=0,
        **kwds,
    )

    if abs(l_k) <= tol:
        msg = f'L-ratio ({r=}, {k=}) denominator is approximately zero.'
        raise ZeroDivisionError(msg)
    t_r = l_r / l_k

    def influence_function(x: _T_x, /) -> _T_x:
        psi_r = if_r(x)
        # cheat a bit to avoid `inf - inf = nan` situations
        psi_k = np.where(np.isinf(psi_r), 0, if_k(x))

        return cast(_T_x, round0((psi_r - t_r * psi_k) / l_k, tol=tol)[()])

    influence_function.__doc__ = (
        f'Theoretical influence function for L-moment ratio with r={_r}, '
        f'k={_k}, and {trim=}.'
    )

    return influence_function
