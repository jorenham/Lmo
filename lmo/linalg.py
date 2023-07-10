__all__ = (
    'sandwich',
    'sh_legendre',
    'sh_jacobi',
    'succession_matrix',
    'trim_matrix',
)

from math import comb, lgamma
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from .typing import AnyInt

T = TypeVar('T', bound=np.number[Any])


def sandwich(
    A: npt.NDArray[np.number[Any]],
    X: npt.NDArray[T | np.number[Any]],
    /,
    dtype: np.dtype[T] | type[T] = np.float_,
) -> npt.NDArray[T]:
    """
    Calculates the "sandwich" matrix product (`A @ X @ A.T`) along the
    specified `X` axis.

    Args:
        A: 2-D array of shape `(s, r)`, the "bread".
        dtype: The data type of the result.
        X: Array of shape `(r, r, ...)`.
    Returns:
        C: Array of shape `(s, s, ...)`.

    See Also:
        - https://wikipedia.org/wiki/Covariance_matrix

    """
    # if X is 1 - d, this is equivalent to: C @ S_b @ C.T
    spec = 'ui, ij..., vj -> uv...'
    return np.einsum(spec, A, X, A, dtype=dtype)  # pyright: ignore


def sh_legendre(k : int, /, dtype: type[T] = np.int_) -> npt.NDArray[T]:
    """
    Shifted Legendre polynomial coefficient matrix $\\widetilde{P}$ of
    shape `(k, k)`, where the $j$-th coefficient of the shifted Legendre
    polynomial of degree $k$ is at $(k, j)$:

    $$
    \\widetilde{p}_{k, j} = (-1)^{k - j} \\binom{k}{j} \\binom{k + j}{j}
    $$

    Useful for transforming probability-weighted moments into L-moments.

    Args:
        k: The size of the matrix, and the max degree of the shifted Legendre
            polynomial.
        dtype:
            Desired output data type, e.g, `numpy.float64`. Default is
            `numpy.int64`.

    Returns:
        P: 2-D array of the lower-triangular square matrix of size $k^2$`.

    Examples:
        Calculate $\\widetilde{P}_{4 \\times 4}$:

        >>> from lmo.linalg import sh_legendre
        >>> sh_legendre(4)
        array([[  1,   0,   0,   0],
               [ -1,   2,   0,   0],
               [  1,  -6,   6,   0],
               [ -1,  12, -30,  20]])

    See Also:
        - https://wikipedia.org/wiki/Legendre_polynomials
        - https://wikipedia.org/wiki/Pascal_matrix

    """
    return sh_jacobi(k, 0, 0).astype(dtype)


def _sh_jacobi_i(k: int, a: int, b: int, dtype: type[T]) -> npt.NDArray[T]:
    out = np.zeros((k, k), dtype=dtype)
    for r in range(k):
        for j in range(r + 1):
            out[r, j] = (
                (-1) ** (r - j)
                * comb(r + a + b + j, j)
                * comb(r + b, r - j)
            )
    return out


def _sh_jacobi_f(k: int, a: float, b: float, dtype: type[T]) -> npt.NDArray[T]:
    out = np.zeros((k, k), dtype=dtype)

    # semi dynamic programming
    lfact_j = np.array([lgamma(j + 1) for j in range(k)])
    lfact_jb = np.array([lgamma(j + b + 1) for j in range(k)])
    lfact_jab = np.array([lgamma(j + a + b + 1) for j in range(k * 2)])

    for r in range(k):
        # log of (r + a + b) to the falling power of a, i.e.
        # lgamma(r + a + b + 1) - lgamma(r + b + 1)
        log_rab_fpow_a = lfact_jab[r] - lgamma(r + b + 1)
        for j in range(r + 1):
            out[r, j] = (-1) ** (r - j) * np.exp(
                lfact_jab[r + j]
                - lfact_jb[j]
                - lfact_j[j]
                - lfact_j[r - j]
                - log_rab_fpow_a
            )
    return out


def sh_jacobi(
    k: AnyInt,
    a: T | int,
    b: T | int,
    /,
    dtype: type[T] | None = None,
) -> npt.NDArray[T | np.int_]:
    """
    Shifted Jacobi polynomial coefficient matrix $\\widetilde{P}^{(a,b)}$ of
    shape `(k, k)`, where the $j$-th coefficient of the shifted Jacobi
    polynomial of degree $k$ is at $(k, j)$:

    The "shift" refers to the change of variables $x \\mapsto 2x - 1$ in the
    (unshifted) Jacobi (or hypergeometric) polynomials.

    The (shifted) Jacobi polynomials $\\widetilde{P}^{(a,b)}$ generalize  the
    (shifted) Legendre polynomials $\\widetilde{P}$:
    $\\widetilde{P}^{(0, 0)} = \\widetilde{P}$

    Notes:
        Analogous to the shifted Legendre polynomials, this implementation


    Args:
        k: The size of the matrix, and the max degree of the polynomial.
        a: The $\\alpha$ parameter, must be $\\ge 0$.
        b: The $\\beta$ parameter, must be $\\ge 0$.
        dtype:
            Desired output data type, e.g, `numpy.float64`. Default is
            `numpy.int64` if `a` and `b` are integers, otherwise `np.float64`.

    Returns:
        P: 2-D array of the lower-triangular square matrix of size $k^2$`.

    Examples:
        Calculate $\\widetilde{P}^{(1, 1)}_{4 \\times 4}$:

        >>> from lmo.linalg import sh_jacobi
        >>> sh_jacobi(4, 1, 1)
        array([[  1,   0,   0,   0],
               [ -2,   4,   0,   0],
               [  3, -15,  15,   0],
               [ -4,  36, -84,  56]])

        Let's compare $\\widetilde{P}^(1, \\pi)_3$ with the scipy Jacobi
        poly1d. This requires manual shifting $x \\mapsto f(x)$,
        with $f(x) = 2x - 1$:

        >>> import numpy as np
        >>> import scipy.special as sc
        >>> f_x = np.poly1d([2, -1])  # f(x) = 2*x + 1
        >>> sc.jacobi(3, 1, np.pi)(f_x)
        poly1d([ 125.80159497, -228.55053774,  128.54584648,  -21.79690371])
        >>> sh_jacobi(4, 1, np.pi)[3]
        array([ -21.79690371,  128.54584648, -228.55053774,  125.80159497])

        Apart from the reversed coefficients of [`numpy.poly1d`][numpy.poly1d]
        (an awkward design choice, but it's fixed in the new
        [`numpy.polynomial`][numpy.polynomial] module.)

    See Also:
        - https://mathworld.wolfram.com/JacobiPolynomial.html
        - [`scipy.special.jacobi`][scipy.special.jacobi]

    """
    if k < 0:
        raise ValueError('m must be >= 0')
    if a < 0:
        raise ValueError('t must be >= 0')
    if b < 0:
        raise ValueError('s must be >= 0')

    _dtype = dtype or np.asarray([a, b]).dtype.type
    if np.issubdtype(_dtype, np.integer):
        return _sh_jacobi_i(int(k), int(a), int(b), _dtype)
    else:
        return _sh_jacobi_f(int(k), float(a), float(b), _dtype)



def succession_matrix(c: npt.NDArray[T], /) -> npt.NDArray[T]:
    """
    A toeplitz-like transformation matrix construction, that prepends $i$
    zeroes to $i$-th row, so that the input shape is mapped from `(n, k)`
    to `(n, k + n)`.

    So all values $i > j \\vee i + j \\ge k$ are zero in the succession matrix.

    Args:
        c: Dense matrix of shape `(n, k)`.

    Returns:
        S: Matrix of shape `(n, k + n)`

    Examples:
        >>> from lmo.linalg import succession_matrix
        >>> succession_matrix(np.arange(1, 9).reshape(4, 2))
        array([[1, 2, 0, 0, 0],
               [0, 3, 4, 0, 0],
               [0, 0, 5, 6, 0],
               [0, 0, 0, 7, 8]])

    """

    n, k = np.atleast_1d(c).shape
    i = np.linspace(0, n - 1, n, dtype=np.int64)

    out = np.zeros((n, n + k - 1), dtype=c.dtype)
    for d in range(k):
        out[i, i + d] = c[:, d]

    return out


def trim_matrix(
    r: int,
    /,
    trim: tuple[int, int],
    dtype: np.dtype[T] | type[T] = np.float_,
) -> npt.NDArray[np.floating[Any]]:
    """
    Linearization of the trimmed L-moment recurrence relations, following
    the (corrected) derivation by Hosking (2007) from the (shifted) Jacobi
    Polynomials.

    This constructs a $r \\times r + t_1 + t_2$ matrix $T^{(t_1, t_2)}$ that
    "trims" conventional L-moments. E.g. the first 3 $(1, 1)$ trimmed
    L-moments can be obtained from the first $3+1+1=5$ (untrimmed) L-moments
    (assuming they exist) with
    `trim_matrix(3, (1, 1)) @ l_moment(x, np.ogrid[:5] + 1)`.

    The big "L" in "L-moment", referring to it being a *Linear* combination of
    order statistics, has been prominently put in the name by Hosking (1990)
    for a good reason. It means that transforming order statistics to
    a bunch of L-moments, can be done using a single matrix multiplication
    (see [`lmo.linalg.sh_legendre`][lmo.linalg.sh_legendre]).
    By exploiting liniarity, it can easily be chained with this trim matrix,
    to obtain a re-usable order-statistics -> trimmed L-moments
    transformation (matrix).

    Note that these linear transformations can be used in exactly the same way
    to e.g. calculate several population TL-moments of some random varianble,
    using nothing but its theoretical probablity-weighted moments (PWMs).

    Args:
        r: The max (trimmed) L-moment order.
        trim: Left- and right-trim orders $(t_1, t_2)$, integers $\\ge 0$.
            If set to (0, 0), the identity matrix is returned.
        dtype: Desired output data type, e.g, `numpy.float64` (default).

    Returns:
        Toeplitz-like matrix of shape $(r, r + t_1 + t_2)$.

    Examples:
        >>> from lmo.linalg import trim_matrix
        >>> trim_matrix(3, (0, 1))
        array([[ 1.        , -1.        ,  0.        ,  0.        ],
               [ 0.        ,  0.75      , -0.75      ,  0.        ],
               [ 0.        ,  0.        ,  0.66666667, -0.66666667]])
        >>> trim_matrix(3, (1, 0))
        array([[1.        , 1.        , 0.        , 0.        ],
               [0.        , 0.75      , 0.75      , 0.        ],
               [0.        , 0.        , 0.66666667, 0.66666667]])

    References:
        - [J.R.M. Hosking (2007) - Some theory and practical uses of trimmed
            L-moments](https://doi.org/10.1016/j.jspi.2006.12.002)
    """

    assert r >= 0

    if r == 0:
        return np.empty((0, 0), dtype=dtype)

    rr = np.linspace(1, r, r, dtype=np.int64)

    t1, t2 = trim
    nc = t1 + t2 - 1 + 2 * rr
    c0 = (t1 + t2 + rr) / nc

    match t1, t2:
        case (0, 0):
            return np.eye(r, dtype=dtype)
        case (0, 1) | (1, 0):
            # (r + 1) / (2 r) * (l_r +/- l_{r+1})
            # = (r + s + t) / (2r + s + t - 1) * (l_r +/- l_{r+1})
            return succession_matrix(np.outer(c0, [1, t1 - t2]))
        case (1, 1):
            # (r + 1)(r + 2) / (2 r (2r + 1)) * (l_r +/- l_{r+2})
            # and (r + 1)(r + 2) / (2 r (2r + 1)) = c0 * (r + 1) / (2 r)
            return succession_matrix(np.outer(c0 * (.5 + .5 / rr), [1, 0, -1]))
        case (s, t) if s < t:
            # ((r+s+t) * _[r+0] - (r+1) * (r+s) * _[r+1] / r) / (2r+s+t-1)
            c1 = -(rr + 1) * (rr + s) / (rr * nc)
            m0 = succession_matrix(np.c_[c0, c1])
            m1 = trim_matrix(r + 1, (s, t - 1), dtype)
            return m0 @ m1
        case (s, t) if s >= t:
            c1 = (rr + 1) * (rr + t) / (rr * nc)
            m0 = succession_matrix(np.c_[c0, c1])
            m1 = trim_matrix(r + 1, (s - 1, t), dtype)
            return m0 @ m1
        case _ as wtf:
            assert False, wtf
