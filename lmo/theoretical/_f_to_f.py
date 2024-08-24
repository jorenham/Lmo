from __future__ import annotations

from typing import TYPE_CHECKING, Concatenate, ParamSpec, cast

import numpy as np

import lmo.typing.np as lnpt
import lmo.typing.scipy as lspt


if TYPE_CHECKING:
    from collections.abc import Callable


__all__: list[str] = ['cdf_from_ppf']


_Tss = ParamSpec('_Tss')


def cdf_from_ppf(
    ppf: Callable[Concatenate[float, _Tss], lnpt.Float | float],
    /,
) -> Callable[Concatenate[float, _Tss], float]:
    """
    Numerical inversion of the PPF.

    Note:
        This function isn't vectorized.
    """
    from scipy.optimize import (
        root_scalar,  # pyright: ignore[reportUnknownVariableType]
    )

    def cdf(x: float, /, *args: _Tss.args, **kwds: _Tss.kwargs) -> float:
        if np.isnan(x):
            return np.nan
        if x <= ppf(0, *args, **kwds):
            return 0
        if x >= ppf(1, *args, **kwds):
            return 1

        def _ppf_to_solve(p: float) -> lnpt.Float | float:
            return ppf(p, *args, **kwds) - x

        result = cast(
            lspt.RootResult,
            root_scalar(_ppf_to_solve, bracket=[0, 1], method='brentq'),
        )
        return result.root

    return cdf
