# ruff: noqa: N803
"""Linear algebra and linearized orthogonal polynomials."""

__all__ = (
    'sandwich',
    'pascal',
    'ir_pascal',
    'sh_legendre',
    'sh_jacobi',
    'succession_matrix',
    'trim_matrix',
)

import sys
from math import comb, lgamma
from typing import Any, TypeVar, cast

if sys.version_info < (3, 11):
    from typing_extensions import assert_never
else:
    from typing import assert_never

import numpy as np
import numpy.typing as npt

from .typing import AnyFloat, AnyInt

T = TypeVar('T', bound=np.object_ | np.integer[Any] | np.floating[Any])


def sandwich(
    A: npt.NDArray[np.number[Any]],
    X: npt.NDArray[T | np.number[Any]],
    /,
    dtype: np.dtype[T] | type[T] = np.float64,
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


def pascal(
    k: int,
    /,
    dtype: np.dtype[T] | type[T] = np.int64,
    *,
    inv: bool = False,
) -> npt.NDArray[T]:
    r"""
    Construct the lower-diagonal Pascal matrix $L_{k \times k$}$, or its matrix
    inverse $L^{-1}$.

    $$
    \begin{align}
    L_{ij} &= \binom{i}{j} \\
    L^{-1}_{ij} &= (-1)^{i - j} L_{ij}
    \end{align}
    $$

    Implemented using recursion, unlike the slow naive implementation from the
    equivalent [`scipy.linalg.pascal`][scipy.linalg.pascal] and
    [`scipy.linalg.invpascal`][scipy.linalg.invpascal] functions using
    `kind='lower'`.
    By using the binomial recurrence relation, assuming $0 < j < i$,
    $\binom{i}{j} = \frac{i}{j} \binom{i-1}{j-1}$, the following recursive
    definition is obtained:

    $$
    L_{ij} =
    \begin{cases}
        0 & \text{if } i < j \text{,} \\
        1 & \text{if } i = j \vee j = 0 \text{, and} \\
        (i \, L_{i-1,\, j-1}) / j & \text{otherwise.}
    \end{cases}
    $$

    Examples:
        >>> import numpy as np
        >>> from lmo.linalg import pascal
        >>> pascal(4, dtype=np.int_)
        array([[1, 0, 0, 0],
               [1, 1, 0, 0],
               [1, 2, 1, 0],
               [1, 3, 3, 1]])
        >>> pascal(4, dtype=np.int_, inv=True)
        array([[ 1,  0,  0,  0],
               [-1,  1,  0,  0],
               [ 1, -2,  1,  0],
               [-1,  3, -3,  1]])
        >>> np.rint(np.linalg.inv(pascal(4))).astype(int)
        array([[ 1,  0,  0,  0],
               [-1,  1,  0,  0],
               [ 1, -2,  1,  0],
               [-1,  3, -3,  1]])

        Now, let's compare with scipy:

        >>> from scipy.linalg import invpascal
        >>> invpascal(4, kind='lower').astype(int)
        array([[ 1,  0,  0,  0],
               [-1,  1,  0,  0],
               [ 1, -2,  1,  0],
               [-1,  3, -3,  1]])
    """
    assert k >= 0
    out = np.zeros((k, k), dtype=dtype)
    if k == 0:
        return out

    out[:, 0] = 1
    if inv:
        # 1337 matrix inversion
        out[1::2, 0] = -1

    jj = np.arange(1, k)
    for i in jj:
        out[i, 1 : i + 1] = i * out[i - 1, :i] // jj[:i]

    return out


def ir_pascal(
    k: int,
    /,
    dtype: np.dtype[T] | type[T] = np.float64,
) -> npt.NDArray[np.float64]:
    r"""
    Inverse regulatized lower-diagonal Pascal matrix,
    $\bar{L}_{ij} = L^{-1}_ij / i$.

    Used to linearly combine order statistics order statistics into L-moments.
    """
    # use native ints to reduce the effect of over-/underflows
    dtype_native = k > 62
    _dtype = np.object_ if dtype_native else np.int64

    p = pascal(k, dtype=_dtype, inv=True)
    out = p / np.arange(1, k + 1, dtype=_dtype)[:, None]  # type: ignore

    return np.asarray(out, dtype)


def _sh_jacobi_i(
    k: int,
    a: int,
    b: int,
    /,
    dtype: np.dtype[T] | type[T],
) -> npt.NDArray[T]:
    out = np.zeros((k, k), dtype=dtype)
    for r in range(k):
        for j in range(r + 1):
            out[r, j] = (
                (-1) ** (r - j) * comb(r + a + b + j, j) * comb(r + b, r - j)
            )
    return out


def _sh_jacobi_f(
    k: int,
    a: float,
    b: float,
    /,
    dtype: np.dtype[T] | type[T],
) -> npt.NDArray[T]:
    out = np.zeros((k, k), dtype=dtype)

    # semi dynamic programming
    lfact_j = np.array([lgamma(j + 1) for j in range(k)])
    lfact_jb = np.array([lgamma(j + b + 1) for j in range(k)])
    lfact_jab = np.array([lgamma(j + a + b + 1) for j in range(k * 2)])

    for r in range(k):
        # log of (r + a + b) to the falling power of a, i.e.
        # `lgamma(r + a + b + 1) - lgamma(r + b + 1)`
        log_rab_fpow_a = lfact_jab[r] - lgamma(r + b + 1)
        for j in range(r + 1):
            out[r, j] = (-1) ** (r - j) * np.exp(
                lfact_jab[r + j]
                - lfact_jb[j]
                - lfact_j[j]
                - lfact_j[r - j]
                - log_rab_fpow_a,
            )
    return out


def sh_legendre(
    k: int,
    /,
    dtype: np.dtype[T] | type[T] = np.int64,
) -> npt.NDArray[T]:
    r"""
    Shifted Legendre polynomial coefficient matrix $\widetilde{P}$ of
    shape `(k, k)`.

    The $j$-th coefficient of the shifted Legendre polynomial of degree $k$ is
    at $(k, j)$:

    $$
    \widetilde{p}_{k, j} = (-1)^{k - j} \binom{k}{j} \binom{k + j}{j}
    $$

    Useful for transforming probability-weighted moments into L-moments.

    Danger:
        For $k \ge 29$, **all** 64-bits dtypes (default is int64) will
        overflow, which results in either an `OverflowError` (if you're
        lucky), or will give incorrect results.
        Similarly, all 32-bits dtypes (e.g. `np.int_` on Windows) already
        overflow when $k \ge 16$.

        **This is not explicitly checked** -- so be sure to select the right
        `dtype` depending on `k`.

        One option is to use `dtype=np.object_`, which will use
        Python-native `int`. However, this is a lot slower, and is likely to
        fail. For instance, when multiplied together with some `float64`
        array, a `TypeError` is raised.

    Args:
        k: The size of the matrix, and the max degree of the shifted Legendre
            polynomial.
        dtype:
            Desired output data type, e.g, `numpy.float64`. Must be signed.
            The default is [`numpy.int64`][numpy.int64].

    Returns:
        P: 2-D array of the lower-triangular square matrix of size $k^2$`.

    Examples:
        Calculate $\widetilde{P}_{4 \times 4}$:

        >>> from lmo.linalg import sh_legendre
        >>> sh_legendre(4, dtype=int)
        array([[  1,   0,   0,   0],
               [ -1,   2,   0,   0],
               [  1,  -6,   6,   0],
               [ -1,  12, -30,  20]])

    See Also:
        - https://wikipedia.org/wiki/Legendre_polynomials
        - https://wikipedia.org/wiki/Pascal_matrix
    """
    return _sh_jacobi_i(k, 0, 0, dtype=dtype)


def sh_jacobi(
    k: AnyInt,
    a: AnyFloat,
    b: AnyFloat,
    /,
    dtype: np.dtype[T] | type[T] | None = None,
) -> npt.NDArray[T | np.int64]:
    r"""
    Shifted Jacobi polynomial coefficient matrix $\widetilde{P}^{(a,b)}$ of
    shape `(k, k)`.

    The $j$-th coefficient of the shifted Jacobi polynomial of degree $k$ is
    at $(k, j)$:

    The "shift" refers to the change of variables $x \mapsto 2x - 1$ in the
    (unshifted) Jacobi (or hypergeometric) polynomials.

    The (shifted) Jacobi polynomials $\widetilde{P}^{(a,b)}$ generalize  the
    (shifted) Legendre polynomials $\widetilde{P}$:
    $\widetilde{P}^{(0, 0)} = \widetilde{P}$

    Args:
        k: The size of the matrix, and the max degree of the polynomial.
        a: The $\alpha$ parameter, must be $\ge 0$.
        b: The $\beta$ parameter, must be $\ge 0$.
        dtype:
            Desired output data type, e.g, `numpy.float64`. Default is
            `numpy.int64` if `a` and `b` are integers, otherwise `np.float64`.

    Returns:
        P: 2-D array of the lower-triangular square matrix of size $k^2$`.

    Examples:
        Calculate $\widetilde{P}^{(1, 1)}_{4 \times 4}$:

        >>> from lmo.linalg import sh_jacobi
        >>> sh_jacobi(4, 1, 1, dtype=int)
        array([[  1,   0,   0,   0],
               [ -2,   4,   0,   0],
               [  3, -15,  15,   0],
               [ -4,  36, -84,  56]])

        Let's compare $\widetilde{P}^(1, \pi)_3$ with the scipy Jacobi
        poly1d. This requires manual shifting $x \mapsto f(x)$,
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
    _k, _a, _b = int(k), float(a), float(b)
    if _k < 0 or _a < 0 or _b < 0:
        msg = 'k, a, and b must be >= 0'
        raise ValueError(msg)

    _dtype = dtype or np.asarray([a, b]).dtype.type
    if np.issubdtype(_dtype, np.integer) or np.issubdtype(_dtype, np.bool_):
        return _sh_jacobi_i(_k, int(a), int(b), dtype=_dtype)

    return _sh_jacobi_f(_k, float(a), float(b), dtype=_dtype)


def succession_matrix(c: npt.NDArray[T], /) -> npt.NDArray[T]:
    r"""
    A toeplitz-like transformation matrix construction, that prepends $i$
    zeroes to $i$-th row, so that the input shape is mapped from `(n, k)`
    to `(n, k + n)`.

    So all values $i > j \vee i + j \ge k$ are zero in the succession matrix.

    Args:
        c: Dense matrix of shape `(n, k)`.

    Returns:
        S: Matrix of shape `(n, k + n)`

    Examples:
        >>> from lmo.linalg import succession_matrix
        >>> c = np.arange(1, 9).reshape(4, 2)
        >>> c
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
        >>> succession_matrix(c)
        array([[1, 2, 0, 0, 0],
               [0, 3, 4, 0, 0],
               [0, 0, 5, 6, 0],
               [0, 0, 0, 7, 8]])
    """
    _c = np.atleast_2d(c)

    n, k = _c.shape
    i = np.arange(n)

    out = np.zeros((n, n + k - 1), dtype=_c.dtype)
    for d in range(k):
        out[i, i + d] = _c[:, d]
    return out


def trim_matrix(
    r: int,
    /,
    trim: tuple[int, int],
    dtype: np.dtype[T] | type[T] = np.float64,
) -> npt.NDArray[T]:
    r"""
    Linearization of the trimmed L-moment recurrence relations, following
    the (corrected) derivation by Hosking (2007) from the (shifted) Jacobi
    Polynomials.

    This constructs a $r \times r + t_1 + t_2$ matrix $T^{(t_1, t_2)}$ that
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
    to obtain a reusable order-statistics -> trimmed L-moments
    transformation (matrix).

    Note that these linear transformations can be used in exactly the same way
    to e.g. calculate several population TL-moments of some random varianble,
    using nothing but its theoretical probablity-weighted moments (PWMs).

    Args:
        r: The max (trimmed) L-moment order.
        trim: Left- and right-trim orders $(t_1, t_2)$, integers $\ge 0$.
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

    rr = np.arange(1, r + 1)

    t1, t2 = trim
    nc = t1 + t2 - 1 + 2 * rr
    c0 = (t1 + t2 + rr) / nc

    match t1, t2:
        case (0, 0):
            out = np.eye(r, dtype=dtype)
        case (0, 1) | (1, 0):
            # (r + 1) / (2 r) * (l_r +/- l_{r+1})
            # = (r + s + t) / (2r + s + t - 1) * (l_r +/- l_{r+1})
            out = succession_matrix(np.outer(c0, [1, t1 - t2]))
        case (1, 1):
            # (r + 1)(r + 2) / (2 r (2r + 1)) * (l_r +/- l_{r+2})
            # and (r + 1)(r + 2) / (2 r (2r + 1)) = c0 * (r + 1) / (2 r)
            out = succession_matrix(
                np.outer(c0 * (0.5 + 0.5 / rr), [1, 0, -1]),
            )
        case (s, t) if s < t:
            # ((r+s+t) * _[r+0] - (r+1) * (r+s) * _[r+1] / r) / (2r+s+t-1)
            c1 = -(rr + 1) * (rr + s) / (rr * nc)
            m0 = succession_matrix(np.c_[c0, c1])
            m1 = trim_matrix(r + 1, (s, t - 1), dtype)
            out = m0 @ m1
        case (s, t) if s >= t:
            c1 = (rr + 1) * (rr + t) / (rr * nc)
            m0 = succession_matrix(np.c_[c0, c1])
            m1 = trim_matrix(r + 1, (s - 1, t), dtype)
            out = m0 @ m1
        case (int(), int()):
            msg = 'trim values must be non-negative'
            raise ValueError(msg)
        case _ as wtf:  # type: ignore [reportUnnecessaryComparison]
            assert_never(wtf)

    return cast(npt.NDArray[T], out)
