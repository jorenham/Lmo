__all__ = 'p_weights', 'l0_weights', 'l_weights'

from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .linalg import sh_legendre, hosking_jacobi


T = TypeVar('T', bound=npt.NBitBase)

def p_weights(
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
        >>> import lmo.weights
        >>> lmo.weights.p_weights(4, 5)
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


def l0_weights(
    r: int,
    n: int,
    /,
    dtype: type[np.floating[T]] | np.dtype[np.floating[T]] = np.float_,
    *,
    enforce_symmetry: bool = True,
) -> npt.NDArray[np.floating[T]]:
    """
    Efficiently calculates the projection matrix $P = [p_{k, i}]_{r \\times n}$
    for the order statistics $x_{i:n}$.
    This way, the $1, 2, ..., r$-th order sample L-moments of some sample vector
    $x$, can be estimated with `np.sort(x) @ l_weights(len(x), r)`.

    Parameters:
        r: The amount of orders to evaluate, i.e. $k = 1, \\dots, r$.
        n: Sample count.
        dtype: Desired output floating data type.

    Other parameters:
        enforce_symmetry:
            If set to False, disables symmetry-based numerical noise correction.

    Returns:
        P_r: 2-D array of shape `(r, n)`.

    Examples:
        >>> import lmo.weights
        >>> lmo.weights.l0_weights(3, 4)
        array([[ 0.25      ,  0.25      ,  0.25      ,  0.25      ],
               [-0.25      , -0.08333333,  0.08333333,  0.25      ],
               [ 0.25      , -0.25      , -0.25      ,  0.25      ]])
        >>> _ @ [-1, 0, 1 / 2, 3 / 2]
        array([0.25      , 0.66666667, 0.        ])

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)

    """
    P_r = np.empty((r, n), dtype)

    if r == 0:
        return P_r

    np.matmul(sh_legendre(r), p_weights(r, n, dtype), out=P_r)

    if enforce_symmetry:
        # enforce rotational symmetry of even orders `r = 2, 4, ...`, naturally
        # centering them around 0
        for k in range(2, r + 1, 2):
            p_k: npt.NDArray[np.floating[T]] = P_r[k - 1]

            med = 0.0
            pk_neg, pk_pos = p_k < med, p_k > med
            # n_neg, n_pos = pk_neg.sum(), pk_pos.sum()
            n_neg, n_pos = np.count_nonzero(pk_neg), np.count_nonzero(pk_pos)

            # attempt to correct 1-off asymmetry
            if abs(n_neg - n_pos) == 1:
                if n % 2:
                    # balance the #negative and #positive for odd `n` by
                    # ignoring the center
                    mid = (n - 1) // 2
                    pk_neg[mid] = pk_pos[mid] = False
                    # n_neg, n_pos = pk_neg.sum(), pk_pos.sum()
                    n_neg = np.count_nonzero(pk_neg)
                    n_pos = np.count_nonzero(pk_pos)
                else:
                    # if one side is half of n, set the other to it's negation
                    mid = n // 2
                    if n_neg == mid:
                        pk_pos = ~pk_neg
                        n_pos = n_neg
                    elif n_pos == mid:
                        pk_neg = ~pk_pos
                        n_neg = n_pos

            # attempt to correct any large asymmetry offsets
            # and don't worry; median is O(n)
            if abs(n_neg - n_pos) > 1 and (med := np.median(p_k)):
                pk_neg, pk_pos = p_k < med, p_k > med
                n_neg = np.count_nonzero(pk_neg)
                n_pos = np.count_nonzero(pk_pos)

            if n_neg == n_pos:
                # it's pretty rare that this isn't the case
                p_k[pk_neg] = -p_k[pk_pos][::-1]

        # enforce horizontal (axis 1) symmetry for the odd orders (except k=1)
        # and shift to zero mean
        P_r[2::2, :n // 2] = P_r[2::2, :(n - 1) // 2: -1]
        P_r[2::2] -= P_r[2::2].mean(1, keepdims=True)

    return P_r


def l_weights(
    r: int,
    n: int,
    /,
    trim: tuple[int, int] = (0, 0),
    dtype: type[np.floating[T]] | np.dtype[np.floating[T]] = np.float_,
) -> npt.NDArray[np.floating[T]]:
    """
    Projection matrix of the first $r$ (T)L-moments for $n$ samples.
    Uses the recurrence relations from Hosking (2007),

    $$
    (2k + t_1 + t_2 - 1) \\lambda^{(t_1, t_2)}_k
        = (k + t_1 + t_2) \\lambda^{(t_1 - 1, t_2)}_k
        + \\frac{1}{k} (k + 1) (k + t_2) \\lambda^{(t_1 - 1, t_2)}_{k+1}
    $$

    for $t_1 > 0$, and

    $$
    (2k + t_1 + t_2 - 1) \\lambda^{(t_1, t_2)}_k
        = (k + t_1 + t_2) \\lambda^{(t_1, t_2 - 1)}_k
        - \\frac{1}{k} (k + 1) (k + t_1) \\lambda^{(t_1, t_2 - 1)}_{k+1}
    $$

    for $t_2 > 0$.

    Returns:
        P_r: 2-D array of shape `(r, n)`.

    """
    if sum(trim) == 0:
        return l0_weights(r, n, dtype)

    P_r = np.empty((r, n), dtype)

    if r == 0:
        return P_r

    # the k-th TL-(t_1, t_2) weights are a linear combination of L-weights
    # with orders k, ..., k + t_1 + t_2

    np.matmul(
        hosking_jacobi(r, trim),
        l0_weights(r + sum(trim), n),
        out=P_r
    )

    # remove numerical noise from the trimmings, and correct for potential
    # shifts in means
    t1, t2 = trim
    P_r[:, :t1] = P_r[:, n - t2:] = 0
    P_r[1:, t1:n - t2] -= P_r[1:, t1:n - t2].mean(1, keepdims=True)

    return P_r
