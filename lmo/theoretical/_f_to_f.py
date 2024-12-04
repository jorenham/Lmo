from __future__ import annotations

from typing import TYPE_CHECKING, Concatenate, ParamSpec

import numpy as np
import optype.numpy as onp

if TYPE_CHECKING:
    from collections.abc import Callable


__all__: list[str] = ["cdf_from_ppf"]


_Tss = ParamSpec("_Tss")


def cdf_from_ppf(
    ppf: Callable[Concatenate[float, _Tss], onp.ToFloat],
    /,
) -> Callable[Concatenate[onp.ToFloat, _Tss], float]:
    """
    Numerical inversion of the PPF.

    Args:
        ppf:
            Quantile function of a univariate continuous probability distribution with
            a signature like `(float, **Tss) -> float-like`. Must be
            monotonically increasing on `[0, 1]`.

    Returns:
        The inverse of the `ppf` with a signature `(float-like, **Tss) -> float`.

    Note:
        This function isn't vectorized (yet)
    """
    from scipy.optimize import root_scalar

    def cdf(x: onp.ToFloat, /, *args: _Tss.args, **kwds: _Tss.kwargs) -> float:
        if np.isnan(x):
            return np.nan
        if x <= ppf(0, *args, **kwds):
            return 0
        if x >= ppf(1, *args, **kwds):
            return 1

        x = float(x)

        def _ppf_to_solve(p: float, /) -> onp.ToFloat:
            return ppf(p, *args, **kwds) - x

        # TODO(jorenham): https://github.com/jorenham/Lmo/issues/362
        result = root_scalar(_ppf_to_solve, bracket=(0, 1), method="brentq")
        return result.root

    return cdf
