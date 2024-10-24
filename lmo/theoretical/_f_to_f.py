from __future__ import annotations

from typing import TYPE_CHECKING, Concatenate, ParamSpec

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import lmo.typing.np as lnpt


__all__: list[str] = ["cdf_from_ppf"]


_Tss = ParamSpec("_Tss")


def cdf_from_ppf(
    ppf: Callable[Concatenate[float, _Tss], lnpt.Float | float],
    /,
) -> Callable[Concatenate[float, _Tss], float]:
    """
    Numerical inversion of the PPF.

    Note:
        This function isn't vectorized.
    """
    from scipy.optimize import root_scalar

    def cdf(x: float, /, *args: _Tss.args, **kwds: _Tss.kwargs) -> float:
        if np.isnan(x):
            return np.nan
        if x <= ppf(0, *args, **kwds):
            return 0
        if x >= ppf(1, *args, **kwds):
            return 1

        def _ppf_to_solve(p: float) -> lnpt.Float | float:
            return ppf(p, *args, **kwds) - x

        result = root_scalar(_ppf_to_solve, bracket=[0, 1], method="brentq")
        return result.root

    return cdf
