from __future__ import annotations

from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeAlias, overload

import numpy as np

from ._utils import QUAD_LIMIT

__all__ = ["entropy_from_qdf"]


_Tss = ParamSpec("_Tss")
_QDF: TypeAlias = (
    Callable[Concatenate[float, _Tss], float]
    | Callable[Concatenate[float, _Tss], np.floating[Any]]
)


@overload
def entropy_from_qdf(qdf: _QDF[[]], /) -> float: ...
@overload
def entropy_from_qdf(
    qdf: _QDF[_Tss],
    /,
    *args: _Tss.args,
    **kwds: _Tss.kwargs,
) -> float: ...
def entropy_from_qdf(
    qdf: _QDF[_Tss],
    /,
    *args: _Tss.args,
    **kwds: _Tss.kwargs,
) -> float:
    r"""
    Evaluate the (differential / continuous) entropy \( H(X) \) of a
    univariate random variable \( X \), from its *quantile density
    function* (QDF), \( q(u) = \frac{\mathrm{d} F^{-1}(u)}{\mathrm{d} u} \),
    with \( F^{-1} \) the inverse of the CDF, i.e. the PPF / quantile function.

    The derivation follows from the identity \( f(x) = 1 / q(F(x)) \) of PDF
    \( f \), specifically:

    \[
        h(X)
            = \E[-\ln f(X)]
            = \int_\mathbb{R} \ln \frac{1}{f(x)} \mathrm{d} x
            = \int_0^1 \ln q(u) \mathrm{d} u
    \]

    Args:
        qdf ( (float, *Tss.args, **Tss.kwargs) -> float):
            The quantile distribution function (QDF).
        *args (Tss.args):
            Optional additional positional arguments to pass to `qdf`.
        **kwds (Tss.kwargs):
            Optional keyword arguments to pass to `qdf`.

    Returns:
        The differential entropy \( H(X) \).

    See Also:
        - [Differential entropy - Wikipedia
        ](https://wikipedia.org/wiki/Differential_entropy)

    """
    import scipy.integrate as spi

    def ic(p: float, /) -> np.float64:
        return np.log(qdf(p, *args, **kwds))

    return spi.quad(ic, 0, 1, limit=QUAD_LIMIT)[0]
