__all__ = (
    '__version__',

    'l_moment',
    'l_ratio',
    'l_loc',
    'l_scale',
    'l_variation',
    'l_skew',
    'l_kurtosis',

    'l_comoment',
    'l_coratio',
    'l_coloc',
    'l_coscale',
    'l_corr',
    'l_coskew',
    'l_cokurtosis',
)

from typing import Final as _Final

from ._meta import get_version as _get_version

from .univariate import (
    l_moment,
    l_ratio,
    l_loc,
    l_scale,
    l_variation,
    l_skew,
    l_kurtosis,
)
from .multivariate import (
    l_comoment,
    l_coratio,
    l_coloc,
    l_coscale,
    l_corr,
    l_coskew,
    l_cokurtosis,
)


__version__: _Final[str] = _get_version()
