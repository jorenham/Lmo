"""
Robust statistics with trimmed L-moments and L-comoments.
"""

import sys  # noqa: I001
from typing import TYPE_CHECKING, Final

from ._lm import (
    l_loc,
    l_scale,
    l_variation,
    l_skew,
    l_kurtosis,

    l_kurt,
    l_moment,
    l_ratio,
    l_stats,

    l_moment_cov,
    l_ratio_se,
    l_stats_se,

    l_moment_influence,
    l_ratio_influence,

    l_weights,
)
from ._lm_co import (
    l_coloc,
    l_coscale,

    l_corr,
    l_coskew,
    l_cokurtosis,
    l_cokurt,

    l_comoment,
    l_coratio,
    l_costats,
)
from ._meta import get_version as _get_version


if not TYPE_CHECKING:
    # install contrib module extensions
    from .contrib import install as _install

    _install()


if 'pytest' in sys.modules:
    import numpy as np

    if np.__version__.startswith('2.'):
        np.set_printoptions(legacy='1.25')  # pyright: ignore[reportArgumentType]

    del np

__version__: Final[str] = _get_version()
__author__: Final[str] = 'Joren Hammdugolu'
__email__: Final[str] = 'jhammudoglu@gmail.com'
__description__: Final[str] = (
    'Robust statistics with trimmed L-moments and L-comoments.'
)
__all__ = (
    '__version__',

    'l_loc',
    'l_scale',
    'l_variation',
    'l_skew',
    'l_kurtosis',
    'l_kurt',

    'l_moment',
    'l_ratio',
    'l_stats',

    'l_moment_cov',
    'l_ratio_se',
    'l_stats_se',

    'l_moment_influence',
    'l_ratio_influence',

    'l_weights',

    'l_coloc',
    'l_coscale',
    'l_corr',
    'l_coskew',
    'l_cokurtosis',
    'l_cokurt',

    'l_comoment',
    'l_coratio',
    'l_costats',
)
