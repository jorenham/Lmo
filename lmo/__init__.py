"""Lmo: Robust statistics with trimmed L-moments and L-comoments."""

__all__ = (
    '__version__',

    'l_loc',
    'l_scale',
    'l_variation',
    'l_skew',
    'l_kurtosis',

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

    'l_comoment',
    'l_coratio',
    'l_costats',
)

from typing import TYPE_CHECKING, Final

from ._lm import (
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
    l_cokurtosis,
    l_coloc,
    l_comoment,
    l_coratio,
    l_corr,
    l_coscale,
    l_coskew,
    l_costats,
)
from ._meta import get_version as _get_version

if not TYPE_CHECKING:
    # install contrib module extensions
    from .contrib import install as _install

    _install()

__version__: Final[str] = _get_version()
__author__ : Final[str] = 'Joren Hammdugolu'
__email__: Final[str] = 'jhammudoglu@gmail.com'
__description__: Final[str] = (
    'Robust statistics with trimmed L-moments and L-comoments.'
)
