"""L-moments for robust data analysis, inference, and non-parametric statistics."""

import sys  # noqa: I001
from typing import TYPE_CHECKING, Final

from ._lm import (
    l_kurt,
    l_kurtosis,
    l_loc,
    l_moment,
    l_moment_cov,
    l_moment_influence,
    l_ratio,
    l_ratio_influence,
    l_ratio_se,
    l_scale,
    l_skew,
    l_stats,
    l_stats_se,
    l_variation,
    l_weights,
)
from ._lm_co import (
    l_cokurt,
    l_cokurtosis,
    l_coloc,
    l_comoment,
    l_coratio,
    l_corr,
    l_coscale,
    l_coskew,
    l_costats,
)
from . import constants, diagnostic, distributions, errors, linalg, special, theoretical
from ._meta import get_version as _get_version


if not TYPE_CHECKING:
    # install contrib module extensions
    from .contrib import install as _install

    _install()
    del _install


if "pytest" in sys.modules:
    import numpy as np

    if np.__version__.startswith("2."):
        np.set_printoptions(legacy="1.25")  # pyright: ignore[reportArgumentType]

    del np

__version__: Final[str] = _get_version()
__author__: Final[str] = "Joren Hammudoglu"
__email__: Final[str] = "jhammudoglu@gmail.com"
__description__: Final[str] = (
    "Robust statistics with trimmed L-moments and L-comoments."
)

__all__ = ["__version__"]
__all__ += [
    "constants",
    "diagnostic",
    "distributions",
    "errors",
    "linalg",
    "special",
    "theoretical",
]
__all__ += [
    "l_kurt",
    "l_kurtosis",
    "l_loc",
    "l_moment",
    "l_moment_cov",
    "l_moment_influence",
    "l_ratio",
    "l_ratio_influence",
    "l_ratio_se",
    "l_scale",
    "l_skew",
    "l_stats",
    "l_stats_se",
    "l_variation",
    "l_weights",
]

__all__ += [
    "l_cokurt",
    "l_cokurtosis",
    "l_coloc",
    "l_comoment",
    "l_coratio",
    "l_corr",
    "l_coscale",
    "l_coskew",
    "l_costats",
]


def __dir__() -> list[str]:
    return __all__
