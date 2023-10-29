r"""
Order statistics $X_{i:n}$, with $i \in [0, n)$.

Primarily used as an intermediate step for L-moment estimation.

References:
    [H.A. David & H.N. Nagaraja (2004) &ndash; Order statistics
    ](https://books.google.com/books?id=bdhzFXg6xFkC)
"""

import functools
from collections.abc import Sequence
from math import floor
from typing import Any, cast, overload

import numpy as np
import numpy.typing as npt
from scipy.special import betainc, betaln  # type: ignore

from .typing import AnyNDArray


def _weights(
    i: float,
    n: float,
    N: int,  # noqa: N803
    /,
) -> npt.NDArray[np.float64]:
    assert 0 <= i < n <= N

    j = np.arange(floor(i), N)

    return np.r_[
        np.zeros(j[0]),
        np.exp(
            cast(
                float,
                betaln(j + 1, N - j)
                - betaln(i + 1, n - i)
                - betaln(j - i + 1, N - j - (n - i) + 1)
                - np.log(N - n + 1),
            ),
        ),
    ]


_weights_cached = functools.lru_cache(1 << 10)(_weights)


def weights(
    i: float,
    n: float,
    N: int,  # noqa: N803
    /,
    *,
    cached: bool = False,
) -> npt.NDArray[np.float64]:
    r"""
    Compute the linear weights $w_{i:n|j:N}$ for $j = 0, \dots, N-1$.

    The unbiased sample estimator $\mu_{i:n}$ is then derived from

    $$
    E[X_{i:n}] = \sum_{j=0}^{N-1} w_{i:n|j:N} X_{j:N} \, ,
    $$

    where

    $$
    \begin{aligned}
    w_{i:n|j:N}
    &= \binom{j}{i} \binom{N - j - 1}{n - i - 1} / \binom{N}{n} \\
    &= \frac{1}{N - n + 1}
    \frac{
        B(j + 1, N - j)
    }{
        B(i + 1, n - i) B(j - i + 1, N - j - n + i + 1)
    }
    \end{aligned}
    $$

    Here, $B$ denotes the Beta function,
    $B(a,b) = \Gamma(a) \Gamma(b) / \Gamma(a + b)$.

    Notes:
        This function uses "Python-style" 0-based indexing for $i$, instead
        of the conventional 1-based indexing that is generally used in the
        literature.

    Args:
        i: 0-indexed sample (fractional) index, $0 \le i \lt n$. Negative
            indexing is allowed.
        n: Subsample size, optionally fractional, $0 \le n0$
        N: Sample size, i.e. the observation count.
        cached: Cache the result for `(i, n, n0)`. Defaults to False.

    Returns:
        1d array of size $N$ with (ordered) sample weights.

    """
    if i < 0:
        # negative indexing
        i = n + i

    if (i, n) == (0, 1):
        # identity case
        return np.full(N, 1 / N)
    if i >= n:
        # impossible case
        return np.full(N, np.nan)

    return (_weights_cached if cached else _weights)(i, n, N)


@overload
def from_cdf(F: float, i: float, n: float) -> float:  # noqa: N803
    ...


@overload
def from_cdf(
    F: AnyNDArray[np.floating[Any]] | Sequence[float],  # noqa: N803
    i: float,
    n: float,
) -> npt.NDArray[np.float64]:
    ...


def from_cdf(
    F: npt.ArrayLike,  # noqa: N803
    i: float,
    n: float,
) -> float | npt.NDArray[np.float64]:
    r"""
    Transform $F(X)$ to $F_{i:n}(X)$, of the $i$th variate within subsamples
    of size, i.e. $0 \le i \le n - 1$.

    Args:
        F: Scalar or array-like with the returned value of some cdf, i.e.
            $F_X(x) = P(X \le x)$. Must be between 0 and 1.
        i: 0-indexed sample (fractional) index, $0 \le i < n$.
        n: Subsample size, optionally fractional, $0 \le n0$
    """
    p = np.asanyarray(F)
    if not np.all((p >= 0) & (p <= 1)):
        msg = 'F must lie between 0 and 1'
        raise ValueError(msg)

    return betainc(i + 1, n - i, F)  # type: ignore
