"""Numerical methods for finding population L-comoments."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from lmo._poly import eval_sh_jacobi
from lmo._utils import clean_order, clean_trim, round0
from ._f_to_lm import l_moment_from_cdf
from ._utils import l_const, tighten_cdf_support

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lmo.typing as lmt


__all__ = ["l_comoment_from_pdf", "l_coratio_from_pdf"]


_T = TypeVar("_T")
_Pair: TypeAlias = tuple[_T, _T]

_FloatND: TypeAlias = onp.ArrayND[np.float64]

_ToPDF: TypeAlias = Callable[[onp.ArrayND[np.float64]], onp.ToFloat]
_ToCDF: TypeAlias = Callable[[float], onp.ToFloat]


def l_comoment_from_pdf(
    pdf: _ToPDF,
    cdfs: Sequence[_ToCDF],
    r: lmt.ToOrder0D,
    /,
    trim: lmt.ToTrim = 0,
    *,
    supports: Sequence[_Pair[float]] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
) -> _FloatND:
    r"""
    Evaluate the theoretical L-*co*moment matrix of a multivariate probability
    distribution, using the joint PDF
    $f(\vec x) \equiv f(x_1, x_2, \ldots, x_n)$ of random vector $\vec{X}$,
    and the marginal CDFs $F_k$ of its $k$-th random variable.

    The L-*co*moment matrix is defined as

    $$
    \Lambda_{r}^{(s, t)} =
        \left[
            \lambda_{r [ij]}^{(s, t)}
        \right]_{n \times n}
    \;,
    $$

    with elements

    $$
    \begin{align*}
    \lambda_{r [ij]}^{(s, t)}
        &= c^{(s,t)}_r \int_{\mathbb{R^n}}
            x_i \
            u_j^s \
            (1 - u_j)^t \
            \widetilde{P}^{(t, s)}_{r - 1} (u_j) \
            f(\vec{x}) \
            \mathrm{d} \vec{x} \\
        &= c^{(s,t)}_r \, \mathbb{E}_{\vec{X}} \left[
            X_i \
            U_j^s \
            (1 - U_j)^t \
            \widetilde{P}^{(t, s)}_{r - 1}(U_j)
        \right]
        \, ,
    \end{align*}
    $$

    where $U_j = F_j(X_j)$ and $u_j = F_j(x_j)$ denote the (marginal)
    [probability integral transform
    ](https://wikipedia.org/wiki/Probability_integral_transform) of
    $X_j$ and $x_j \sim X_j$.
    Furthermore, $\widetilde{P}^{(\alpha, \beta)}_k$ is a shifted Jacobi
    polynomial, and

    $$
    c^{(s,t)}_r =
        \frac{r + s + t}{r}
        \frac{\B(r,\ r + s + t)}{\B(r + s,\ r + t)} \;
    ,
    $$

    a positive constant.

    For $r \ge 2$, it can also be expressed as

    $$
    \lambda_{r [ij]}^{(s, t)}
        = c^{(s,t)}_r \mathrm{Cov} \left[
            X_i, \;
            U_j^s \
            (1 - U_j)^t \
            \widetilde{P}^{(t, s)}_{r - 1}(U_j)
        \right] \;
        ,
    $$

    and without trim ($s = t = 0$), this simplifies to

    $$
    \lambda_{r [ij]}
        = \mathrm{Cov} \left[
            X_i ,\;
            \widetilde{P}_{r - 1} (U_j)
        \right] \;
        ,
    $$

    with $\tilde{P}_n = \tilde{P}^{(0, 0)}_n$ the shifted Legendre polynomial.
    This last form is precisely the definition introduced by
    Serfling & Xiao (2007).

    Note that the L-comoments along the diagonal, are equivalent to the
    (univariate) L-moments, i.e.

    $$
    \lambda_{r [ii]}^{(s, t)}\big( \vec{X} \big)
        = \lambda_{r}^{(s, t)}\big( X_i \big) \;.
    $$

    Notes:
        At the time of writing, trimmed L-comoments have not been explicitly
        defined in the literature. Instead, the author
        ([@jorenham](https://github.com/jorenham/)) derived it
        by generizing the (untrimmed) L-comoment definition by Serfling &
        Xiao (2007), analogous to the generalization of L-moments
        into TL-moments by Elamir & Seheult (2003).

    Examples:
        Find the TL-coscale matrix of the multivariate Student's t distribution with 4
        degrees of freedom:

        >>> from scipy.stats import multivariate_t
        >>> df = 4
        >>> loc = np.array([0.5, -0.2])
        >>> cov = np.array([[2.0, 0.3], [0.3, 0.5]])
        >>> X = multivariate_t(loc=loc, shape=cov, df=df)

        >>> from scipy.special import stdtr
        >>> std0, std1 = np.sqrt(np.diag(cov))
        >>> cdf0 = lambda x: stdtr(df, (x - loc[0]) / std0)
        >>> cdf1 = lambda x: stdtr(df, (x - loc[1]) / std1)

        >>> tl_cov = l_comoment_from_pdf(X.pdf, (cdf0, cdf1), 2, trim=1)
        >>> tl_cov.round(4)
        array([[0.4893, 0.1468],
               [0.0734, 0.2447]])

        The (Pearson) correlation coefficient can be recovered in several ways:

        >>> cov[0, 1] / (std0 * std1)  # "true" correlation
        0.3
        >>> tl_cov[0, 1] / tl_cov[0, 0]
        0.3
        >>> tl_cov[1, 0] / tl_cov[1, 1]
        0.3

    Args:
        pdf:
            Joint Probability Distribution Function (PDF), that accepts a
            float vector of size $n$, and returns a scalar in $[0, 1]$.
        cdfs:
            Sequence with $n$ marginal CDF's.
        r:
            Non-negative integer $r$ with the L-moment order.
        trim:
            Left- and right- trim, either as a $(s, t)$ tuple with
            $s, t > -1/2$, or $t$ as alias for $(t, t)$.

    Other parameters:
        supports:
            A sequence with $n$ 2-tuples, corresponding to the marginal
            integration limits. Defaults to $[(-\infty, \infty), \dots]$.
        quad_opts:
            Optional dict of options to pass to
            [`scipy.integrate.quad`][scipy.integrate.quad].

    Returns:
        lmbda:
            The population L-*co*moment matrix with shape $n \times n$.

    References:
        - [E. Elamir & A. Seheult (2003) - Trimmed L-moments](
            https://doi.org/10.1016/S0167-9473(02)00250-5)
        - [R. Serfling & P. Xiao (2007) - A Contribution to Multivariate
          L-Moments: L-Comoment
          Matrices](https://doi.org/10.1016/j.jmva.2007.01.008)
    """
    n = len(cdfs)
    limits = supports or [tighten_cdf_support(cdf, None) for cdf in cdfs]

    r_ = clean_order(int(r))
    s, t = clean_trim(trim)

    c = l_const(r_, s, t)

    # def integrand(*xs: float, i: int, j: int) -> float:
    def integrand(*xs: float, i: int, j: int) -> float:
        q_j = cdfs[j](xs[j])
        p_j = eval_sh_jacobi(r_ - 1, t, s, q_j)
        x = np.asarray(xs, dtype=np.float64)
        return c * x[i] * q_j**s * (1 - q_j) ** t * p_j * pdf(x)

    from scipy.integrate import nquad

    l_r = np.zeros((n, n))

    # TODO: parallelize
    for i, j in np.ndindex(l_r.shape):
        if i == j:
            l_r[i, j] = l_moment_from_cdf(
                cdfs[i],
                r_,
                trim=(s, t),
                support=limits[i],
                quad_opts=quad_opts,
            )
        elif r_:
            fn = partial(integrand, i=i, j=j)
            l_r[i, j] = nquad(fn, limits, opts=quad_opts)[0]

    return round0(l_r)


def l_coratio_from_pdf(
    pdf: _ToPDF,
    cdfs: Sequence[_ToCDF],
    r: lmt.ToOrder0D,
    r0: lmt.ToOrder0D = 2,
    /,
    trim: lmt.ToTrim = 0,
    *,
    supports: Sequence[_Pair[float]] | None = None,
    quad_opts: lmt.QuadOptions | None = None,
) -> onp.ArrayND[npc.floating]:
    r"""
    Evaluate the theoretical L-*co*moment ratio matrix of a multivariate
    probability distribution, using the joint PDF $f_{\vec{X}}(\vec{x})$ and
    $n$ marginal CDFs $F_X(x)$ of random vector $\vec{X}$.

    $$
    \tilde \Lambda_{r,r_0}^{(s, t)} =
        \left[
            \left. \lambda_{r [ij]}^{(s, t)} \right/
            \lambda_{r_0 [ii]}^{(s, t)}
        \right]_{n \times n}
    $$

    See Also:
        - [`l_comoment_from_pdf`][lmo.theoretical.l_comoment_from_pdf]
        - [`lmo.l_coratio`][lmo.l_coratio]
    """
    ll_r = l_comoment_from_pdf(
        pdf,
        cdfs,
        r,
        trim=trim,
        supports=supports,
        quad_opts=quad_opts,
    )
    ll_r0 = l_comoment_from_pdf(
        pdf,
        cdfs,
        r0,
        trim=trim,
        supports=supports,
        quad_opts=quad_opts,
    )

    return ll_r / np.expand_dims(ll_r0.diagonal(), -1)
