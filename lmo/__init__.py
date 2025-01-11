"""L-moments for robust data analysis, inference, and non-parmaetric statistics."""

import sys  # noqa: I001
from typing import TYPE_CHECKING, Final

from . import (
    _lm,
    _lm_co,
    constants,
    diagnostic,
    distributions,
    errors,
    linalg,
    special,
    theoretical,
)
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
__author__: Final[str] = "Joren Hammdugolu"
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
__all__ += _lm.__all__
__all__ += _lm_co.__all__


del TYPE_CHECKING, Final, _get_version, _lm, _lm_co, sys
