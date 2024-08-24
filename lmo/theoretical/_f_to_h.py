from __future__ import annotations

from typing import TYPE_CHECKING, Concatenate, ParamSpec, cast, overload

import numpy as np
import scipy.integrate as spi

from ._utils import QUAD_LIMIT


if TYPE_CHECKING:
    from collections.abc import Callable

    import lmo.typing.np as lnpt


__all__ = ['entropy_from_qdf']


_Tss = ParamSpec('_Tss')


@overload
def entropy_from_qdf(
    qdf: Callable[[float], float | lnpt.Float],
    /,
) -> float: ...
@overload
def entropy_from_qdf(
    qdf: Callable[Concatenate[float, _Tss], float | lnpt.Float],
    /,
    *args: _Tss.args,
    **kwds: _Tss.kwargs,
) -> float: ...
def entropy_from_qdf(
    qdf: Callable[Concatenate[float, _Tss], float | lnpt.Float],
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
        qdf ( (float, *Ts, **Ts) -> float):
            The quantile distribution function (QDF).
        *args (*Ts):
            Optional additional positional arguments to pass to `qdf`.
        **kwds (**Ts):
            Optional keyword arguments to pass to `qdf`.

    Returns:
        The differential entropy \( H(X) \).

    See Also:
        - [Differential entropy - Wikipedia
        ](https://wikipedia.org/wiki/Differential_entropy)

    """
    def ic(p: float) -> np.float64:
        return np.log(qdf(p, *args, **kwds))

    return cast(
        float,
        spi.quad(ic, 0, 1, limit=QUAD_LIMIT)[0],  # pyright: ignore[reportUnknownMemberType]
    )
