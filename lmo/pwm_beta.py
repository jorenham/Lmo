r"""
Low-level utility functions for calculating a special case of the
probability-weighted moments (PWM's), $\beta_k = M_{1,k,0}$.

Primarily used as an intermediate step for L-moment estimation.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np
import numpy.typing as npt

from ._utils import ordered


if sys.version_info >= (3, 13):
    from typing import TypeVar, Unpack
else:
    from typing_extensions import TypeVar, Unpack


if TYPE_CHECKING:
    import optype.numpy as onpt

    import lmo.typing.np as lnpt


__all__ = 'weights', 'cov'


_F = TypeVar('_F', bound=np.floating[Any], default=np.float64)
_R = TypeVar('_R', bound=int)
_N = TypeVar('_N', bound=int)

_DType: TypeAlias = np.dtype[_F] | type[_F]


def weights(
    r: _R,
    n: _N,
    /,
    dtype: _DType[_F] = np.float64,
) -> onpt.Array[tuple[_R, _N], _F]:
    r"""
    Probability Weighted moment (PWM) projection matrix $B$ of the
    unbiased estimator for $\beta_k = M_{1,k,0}$ for $k = 0, \dots, r - 1$.

    The PWM's are estimated by linear projection of the sample of order
    statistics, i.e. $b = B x_{i:n}$

    Parameters:
        r: The amount of orders to evaluate, i.e. $k = 0, \dots, r - 1$.
        n: Sample count.
        dtype: Desired output floating data type.

    Returns:
        P_b: Upper-triangular projection matrix of shape `(r, n)`.

    Examples:
        >>> from lmo import pwm_beta
        >>> pwm_beta.weights(4, 5)
        array([[0.2       , 0.2       , 0.2       , 0.2       , 0.2       ],
               [0.        , 0.05      , 0.1       , 0.15      , 0.2       ],
               [0.        , 0.        , 0.03333333, 0.1       , 0.2       ],
               [0.        , 0.        , 0.        , 0.05      , 0.2       ]])

    """
    if not (0 <= r <= n):
        msg = f'expected 0 <= r <= n, got {r=} and {n=}'
        raise ValueError(msg)

    i1 = np.arange(1, n + 1, dtype=dtype)

    w_r = np.zeros((r, n), dtype)
    if w_r.size == 0:
        return w_r

    w_r[0] = 1 / n

    for k in range(1, r):
        w_r[k, k:] = w_r[k - 1, k:] * i1[:-k] / (n - k)

    # the + 0. eliminates negative zeros
    return cast(npt.NDArray[_F], w_r + 0.0)


@overload
def cov(
    a: lnpt.AnyArrayFloat,
    r: _R,
    /,
    axis: None = ...,
    dtype: _DType[_F] = np.float64,
    **kwds: Any,
) -> onpt.Array[tuple[_R, _R], _F]: ...
@overload
def cov(
    a: lnpt.AnyArrayFloat,
    r: _R,
    /,
    axis: int,
    dtype: _DType[_F] = np.float64,
    **kwds: Any,
) -> onpt.Array[tuple[_R, _R, Unpack[tuple[int, ...]]], _F]: ...
def cov(
    a: lnpt.AnyArrayFloat,
    r: int,
    /,
    axis: int | None = None,
    dtype: _DType[_F] = np.float64,
    **kwds: Any,
) -> onpt.Array[Any, _F]:
    r"""
    Distribution-free variance-covariance matrix of the probability weighted
    moment (PWM) point estimates $\beta_k = M_{1,k,0}$, with orders
    $k = 0, \dots, r - 1$.

    Parameters:
        a: 1-D or 2-D array-like with observations.
        r: The amount of orders to evaluate, i.e. $k = 0, \dots, r - 1$.
        axis: The axis along which to calculate the covariance matrices.
        dtype: Desired output floating data type.
        **kwds: Additional keywords to pass to `lmo.stats.ordered`.

    Returns:
        S_b: Variance-covariance matrix/tensor of shape `(r, r)` or (r, r, n)

    See Also:
        - https://wikipedia.org/wiki/Covariance_matrix

    References:
        - [E. Elmamir & A. Seheult (2004) - Exact variance structure of sample
            L-moments](https://doi.org/10.1016/S0378-3758(03)00213-1)
    """
    x = ordered(a, axis=axis, dtype=dtype, **kwds)

    # ensure the samples are "in front" (along axis=0)
    if axis and x.ndim > 1:
        x = np.moveaxis(x, axis, 0)

    n = len(x)
    p_k = weights(r, n, dtype=dtype)

    # beta pwm estimates
    b = p_k @ x if x.ndim == 1 else np.inner(p_k, x.T)
    assert b.shape == (r, *x.shape[1:])

    # the covariance matrix
    s_b = np.empty((r, *b.shape), dtype=dtype)

    # dynamic programming approach for falling factorial ratios:
    # w_ki[k, i] = i^(k) / (n-1)^(k))
    ffact = np.ones((r, n), dtype=dtype)
    for k in range(1, r):
        ffact[k] = ffact[k - 1] * np.linspace((1 - k) / (n - k), 1, n)

    # ensure that at most ffact[..., -k_max] will give 0
    ffact = np.c_[ffact, np.zeros((r, r))]

    spec: str = 'i..., i...'

    # for k == l (variances on the diagonal):
    # sum(
    #     2 * i^(k) * (j-k-1)^(k) * x[i] * x[j]
    #     for j in range(i + 1, n)
    # ) / n^(k+l+2)
    for k in range(r):
        j_k = np.arange(-k, n - k - 1)
        v_ki = np.empty(x.shape, dtype=dtype)
        for i in range(n):
            # sum(
            #     2 * i^(k) * (j-k-1)^(k) * x[i] * x[j]
            #     for j in range(i + 1, n)
            # )
            v_ki[i] = ffact[k, j_k[i:]] * 2 * ffact[k, i] @ x[i + 1 :]

        # (n-k-1)^(k+1)
        denom = n * (n - 2 * k - 1) * ffact[k, n - k - 1]
        m_bb = np.einsum(spec, v_ki, x) / denom  # pyright: ignore[reportUnknownMemberType]
        s_b[k, k] = b[k] ** 2 - m_bb

    # for k != l (actually k > l since symmetric)
    # sum(
    #     ( i^(k) * (j-k-1)^(l) + i^(l) * (j-l-1)^(k) )
    #     * x[i] * x[j]
    #     for j in range(i + 1, n)
    # ) / n^(k+l+2)
    for k, m in zip(*np.tril_indices(r, -1), strict=True):
        j_k = np.arange(-k, n - k - 1, dtype=int)
        j_l = np.arange(-m, n - m - 1, dtype=int)

        v_ki = np.empty(x.shape, dtype=dtype)
        for i in range(n):
            v_ki[i] = (
                ffact[k, i] * ffact[m, j_k[i:]]
                + ffact[m, i] * ffact[k, j_l[i:]]
            ) @ x[i + 1 :]

        # `(n-k-1)^(l+1)`
        denom = n * (n - k - m - 1) * ffact[m, n - k - 1]
        m_bb = np.einsum(spec, v_ki, x) / denom  # pyright: ignore[reportUnknownMemberType]

        # because s_bb.T == s_bb
        s_b[k, m] = s_b[m, k] = b[k] * b[m] - m_bb

    return s_b
