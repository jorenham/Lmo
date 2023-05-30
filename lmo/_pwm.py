"""
Power-Weighted Moment (PWM) estimators.

Primarily used as an intermediate step for L-moment estimation.
"""
__all__ = 'b_weights', 'b_moment_cov'

from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from .stats import order_stats

T = TypeVar('T', bound=np.floating[Any])


def b_weights(
    r: int,
    n: int,
    /,
    dtype: type[np.floating[T]] | np.dtype[np.floating[T]] = np.float_,
) -> npt.NDArray[np.floating[T]]:
    """
    Probability Weighted moment (PWM) projection matrix $B$  of the
    unbiased estimator for $\\beta_{k} = M_{1,k,0}$ for $k = 0, \\dots, r - 1$.

    The PWM's are estimated by linear projection of the sample of order
    statistics, i.e. $b = B x_{i:n}$

    Parameters:
        r: The amount of orders to evaluate, i.e. $k = 0, \\dots, r - 1$.
        n: Sample count.
        dtype: Desired output floating data type.

    Returns:
        P_b: Upper-triangular projection matrix of shape `(r, n)`.

    Examples:
        >>> b_weights(4, 5)
        array([[0.2       , 0.2       , 0.2       , 0.2       , 0.2       ],
               [0.        , 0.05      , 0.1       , 0.15      , 0.2       ],
               [0.        , 0.        , 0.03333333, 0.1       , 0.2       ],
               [0.        , 0.        , 0.        , 0.05      , 0.2       ]])

    """
    if not (0 <= r <= n):
        raise ValueError(f'expected 0 <= r <= n, got {r=} and {n=}')

    # pyright throws a tantrum with numpy.arange for some reason...
    # i1 = np.arange(1, n + 1)
    i1 = np.linspace(1, n, n)

    w_r = np.zeros((r, n), dtype)
    if w_r.size == 0:
        return w_r

    w_r[0] = 1.

    for k in range(1, r):
        w_r[k, k:] = w_r[k - 1, k:] * i1[:-k] / (n - k)

    # the + 0. eliminates negative zeros
    return w_r / n + 0.


def b_moment_cov(
    a: npt.ArrayLike,
    ks: int,
    /,
    axis: int | None = None,
    dtype: np.dtype[T] | type[T] = np.float_,
    **kwargs: Any,
) -> npt.NDArray[T]:
    """
    Distribution-free variance-covariance matrix of the $\\beta$ PWM point
    estimates with orders `k = 0, ..., ks`.

    Returns:
        S_b: Variance-covariance matrix/tensor of shape `(ks, ks, ...)`

    See Also:
        - https://wikipedia.org/wiki/Covariance_matrix

    References:
        - [E. Elmamir & A. Seheult (2004) - Exact variance structure of sample
            L-moments](https://doi.org/10.1016/S0378-3758(03)00213-1)

    """
    x = order_stats(a, axis=axis, dtype=dtype, **kwargs)

    # ensure the samples are "in front" (along axis=0)
    if axis and x.ndim > 1:
        x = np.moveaxis(x, axis, 0)

    n = len(x)
    P_k = b_weights(ks, n, dtype=dtype)

    # beta pwm estimates
    b = P_k @ x if x.ndim == 1 else np.inner(P_k, x.T)
    assert b.shape == (ks, *x.shape[1:])

    # the covariance matrix
    S_b = np.empty((ks, *b.shape), dtype=dtype)

    # dynamic programming approach for falling factorial ratios:
    # w_ki[k, i] = i^(k) / (n-1)^(k))
    ffact = np.ones((ks, n), dtype=dtype)
    for k in range(1, ks):
        ffact[k] = ffact[k - 1] * np.linspace((1 - k) / (n - k), 1, n)

    # ensure that at most ffact[..., -k_max] will give 0
    ffact = np.c_[ffact, np.zeros((ks, ks))]

    spec = 'i..., i...'

    # for k == l (variances on the diagonal):
    # sum(
    #     2 * i^(k) * (j-k-1)^(k) * x[i] * x[j]
    #     for j in range(i + 1, n)
    # ) / n^(k+l+2)
    for k in range(ks):
        j_k = np.arange(-k, n - k - 1)  # pyright: ignore
        v_ki = np.empty(x.shape, dtype=dtype)
        for i in range(n):
            # sum(
            #     2 * i^(k) * (j-k-1)^(k) * x[i] * x[j]
            #     for j in range(i + 1, n)
            # )
            v_ki[i] = ffact[k, j_k[i:]] * 2 * ffact[k, i] @ x[i + 1:]

        # (n-k-1)^(k+1)
        denom = n * (n - 2 * k - 1) * ffact[k, n - k - 1]
        m_bb = np.einsum(spec, v_ki, x) / denom  # pyright: ignore
        S_b[k, k] = b[k]**2 - m_bb

    # for k != l (actually k > l since symmetric)
    # sum(
    #     ( i^(k) * (j-k-1)^(l) + i^(l) * (j-l-1)^(k) )
    #     * x[i] * x[j]
    #     for j in range(i + 1, n)
    # ) / n^(k+l+2)
    for k, m in zip(*np.tril_indices(ks, -1)):
        j_k: npt.NDArray[np.int_] = np.arange(-k, n - k - 1)  # pyright: ignore
        j_l: npt.NDArray[np.int_] = np.arange(-m, n - m - 1)  # pyright: ignore

        v_ki = np.empty(x.shape, dtype=dtype)
        for i in range(n):
            v_ki[i] = (
                ffact[k, i] * ffact[m, j_k[i:]] +
                ffact[m, i] * ffact[k, j_l[i:]]
            ) @ x[i + 1:]

        # (n-k-1)^(l+1)
        denom = n * (n - k - m - 1) * ffact[m, n - k - 1]
        m_bb = np.einsum(spec, v_ki, x) / denom  # pyright: ignore

        # because s_bb.T == s_bb
        S_b[k, m] = S_b[m, k] = b[k] * b[m] - m_bb

    return S_b
