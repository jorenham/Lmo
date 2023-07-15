__all__ = (
    '__version__',

    'l_weights',
    'l_moment',
    'l_moment_cov',
    'l_ratio',
    'l_ratio_se',
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

from ._lm import (
    l_weights,
    l_moment,
    l_moment_cov,
    l_ratio,
    l_ratio_se,
    l_loc,
    l_scale,
    l_variation,
    l_skew,
    l_kurtosis,
)
from ._lm_co import (
    l_comoment,
    l_coratio,
    l_coloc,
    l_coscale,
    l_corr,
    l_coskew,
    l_cokurtosis,
)

__version__: _Final[str] = _get_version()
