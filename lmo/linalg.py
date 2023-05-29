__all__ = 'sh_legendre', 'succession_matrix', 'tl_jacobi'

from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

_T = TypeVar('_T', bound=npt.NBitBase)
_N = TypeVar('_N', bound=np.number[Any])


def sh_legendre(
    m : int,
    /,
    dtype: type[_N] = np.int_,
) -> npt.NDArray[_N]:
    """
    Shifted legendre polynomial coefficient matrix $P^*$ of shape `(m, m)`,
    where the $k$-th coefficient of the shifted Legendre polynomial of
    degree $n$ is at $(n, k)$:

    $$
    p^*_{n, k} = (-1)^{n - k} \\binom{n}{k} \\binom{n + k}{k}
    $$

    Implemented as the elementwise product of of the symmetric Pascal matrix
    with the inverse of the lower Pascal matrix.

    Useful for transforming probability-weighted moments into L-moments.

    Args:
        m: The size of the matrix, and the max degree of the shifted Legendre
            polynomial.
        dtype:
            Desired output data type, e.g, `numpy.float64`. Default is
            `numpy.int64`.

    Returns:
        P: 2-D array of the lower-triangular square matrix of size $m$`.

    Examples:
        Calculate $P_{4 \\times 4}$:

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
    if m < 0:
        raise ValueError
    if m == 0:
        return np.zeros((0, 0), dtype=dtype)

    # Calculate both the lower inverse- and symmetric Pascal matrices
    lp = np.zeros((m, m), dtype=dtype)
    p2 = np.ones((m, m), dtype=dtype)

    lp[0, 0] = 1
    for i in range(1, m):
        lp[i, 0] = (-1) ** i
        for j in range(1, m):
            lp[i, j] = lp[i - 1, j - 1] - lp[i - 1, j]
            p2[i, j] = p2[i - 1, j] + p2[i, j - 1]

    np.multiply(lp, p2, out=lp)
    return lp


def succession_matrix(c: npt.NDArray[_N], /) -> npt.NDArray[_N]:
    # TODO: docsstring + tests

    n, k = np.atleast_1d(c).shape
    i = np.linspace(0, n - 1, n, dtype=np.int64)

    out = np.zeros((n, n + k - 1), dtype=c.dtype)
    for d in range(k):
        out[i, i + d] = c[:, d]

    return out


def tl_jacobi(
    r: int,
    /,
    trim: tuple[int, int],
    dtype: type[_N] = np.float_,
) -> npt.NDArray[np.floating[Any]]:
    # TODO: docsstring + tests

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
            m1 = tl_jacobi(r + 1, (s, t - 1), dtype)
            return m0 @ m1
        case (s, t) if s >= t:
            c1 = (rr + 1) * (rr + t) / (rr * nc)
            m0 = succession_matrix(np.c_[c0, c1])
            m1 = tl_jacobi(r + 1, (s - 1, t), dtype)
            return m0 @ m1
        case _ as wtf:
            assert False, wtf
