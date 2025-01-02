from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from lmo._poly import eval_sh_jacobi
from lmo._utils import clean_order, clean_trim, round0
from ._f_to_lm import l_moment_from_cdf
from ._utils import ALPHA, l_const, tighten_cdf_support

if TYPE_CHECKING:
    import lmo.typing as lmt


__all__ = ["l_moment_influence_from_cdf", "l_ratio_influence_from_cdf"]


_T = TypeVar("_T")
_Pair: TypeAlias = tuple[_T, _T]

_FloatND: TypeAlias = onp.ArrayND[npc.floating]


class _Fn1(Protocol):
    @overload
    def __call__(self, x: onp.ToFloat, /) -> float: ...
    @overload
    def __call__(self, x: onp.ToFloatND, /) -> _FloatND: ...


###


def l_moment_influence_from_cdf(
    cdf: _Fn1,
    r: lmt.ToOrder0D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    l_moment: onp.ToFloat | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
    tol: float = 1e-8,
) -> _Fn1:
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
    r_ = clean_order(int(r))
    if r_ == 0:

        @overload
        def influence0(x: onp.ToFloat, /) -> float: ...
        @overload
        def influence0(x: onp.ToFloatND, /) -> _FloatND: ...
        def influence0(x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
            """
            L-moment Influence Function for `r=0`.

            Args:
                x: Scalar or array-like of sample observarions.

            Returns:
                out
            """
            return 0.0 if np.isscalar(x) else np.zeros_like(x)

        return influence0

    s, t = clean_trim(trim)

    if l_moment is None:
        lm = l_moment_from_cdf(
            cdf,
            r_,
            trim=(s, t),
            support=support,
            quad_opts=quad_opts,
            alpha=alpha,
        )
    else:
        lm = l_moment

    a, b = support or tighten_cdf_support(cdf, support)
    c = l_const(r_, s, t)

    @overload
    def influence(x: onp.ToFloat, /) -> float: ...
    @overload
    def influence(x: onp.ToFloatND, /) -> _FloatND: ...
    def influence(x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        x_ = np.asanyarray(x, np.float64)
        q = np.piecewise(
            x_,
            [x_ <= a, (x_ > a) & (x_ < b), x_ >= b],
            [0, cdf, 1],
        )
        w = round0(c * q**s * (1 - q) ** t, tol)

        # cheat a bit and replace 0 * inf by 0, ensuring convergence if s or t
        alpha = w * eval_sh_jacobi(r_ - 1, t, s, q) * np.where(w, x_, 0)
        out = alpha - lm
        return round0(out.item() if out.ndim == 0 and np.isscalar(x) else out, tol)

    influence.__doc__ = (
        f"Theoretical influence function for L-moment with {r=} and {trim=}."
    )

    return influence


def l_ratio_influence_from_cdf(
    cdf: _Fn1,
    r: lmt.ToOrder0D,
    k: lmt.ToOrder0D = 2,
    /,
    trim: lmt.ToTrim = 0,
    *,
    support: _Pair[float] | None = None,
    l_moments: _Pair[onp.ToFloat] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
    alpha: float = ALPHA,
    tol: float = 1e-8,
) -> _Fn1:
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
    r_, k_ = clean_order(int(r)), clean_order(int(k))

    kwds: dict[str, Any] = {"support": support, "quad_opts": quad_opts}

    if l_moments is None:
        l_r, l_k = l_moment_from_cdf(cdf, [r_, k_], trim=trim, alpha=alpha, **kwds)
    else:
        l_r, l_k = l_moments

    if_r = l_moment_influence_from_cdf(cdf, r_, trim, l_moment=l_r, tol=0, **kwds)
    if_k = l_moment_influence_from_cdf(cdf, k_, trim, l_moment=l_k, tol=0, **kwds)

    if abs(l_k) <= tol:
        msg = f"L-ratio ({r=}, {k=}) denominator is approximately zero."
        raise ZeroDivisionError(msg)

    t_r = l_r / l_k

    @overload
    def influence(x: onp.ToFloat, /) -> float: ...
    @overload
    def influence(x: onp.ToFloatND, /) -> _FloatND: ...
    def influence(x: onp.ToFloat | onp.ToFloatND, /) -> float | _FloatND:
        psi_r = if_r(x)
        # cheat a bit to avoid `inf - inf = nan` situations
        psi_k = np.where(np.isinf(psi_r), 0, if_k(x))
        out = (psi_r - t_r * psi_k) / l_k
        return round0(out.item() if out.ndim == 0 and np.isscalar(x) else out, tol=tol)

    influence.__doc__ = (
        f"Theoretical influence function for L-moment ratio with r={r_}, "
        f"k={k_}, and {trim=}."
    )

    return influence
