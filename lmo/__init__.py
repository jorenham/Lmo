__all__ = (
    '__version__',

    'weights',

    'l_moment',
    'l_ratio',
    'l_loc',
    'l_scale',
    'l_skew',
    'l_kurt',

    'tl_moment',
    'tl_ratio',
    'tl_loc',
    'tl_scale',
    'tl_skew',
    'tl_kurt',

    'l_comoment',
    'l_coratio',
    'l_coloc',
    'l_coscale',
    'l_corr',
    'l_coskew',
    'l_cokurt',

    'tl_comoment',
    'tl_coratio',
    'tl_coloc',
    'tl_coscale',
    'tl_corr',
    'tl_coskew',
    'tl_cokurt',
)

from typing import Final as _Final

from ._meta import get_version as _get_version

from . import weights
from .univariate import (
    tl_moment, l_moment,
    tl_ratio, l_ratio,
    tl_loc, l_loc,
    tl_scale, l_scale,
    tl_skew, l_skew,
    tl_kurt, l_kurt,
)
from .multivariate import (
    tl_comoment, l_comoment,
    tl_coratio, l_coratio,
    tl_coloc, l_coloc,
    tl_coscale, l_coscale,
    tl_corr, l_corr,
    tl_coskew, l_coskew,
    tl_cokurt, l_cokurt,
)


__version__: _Final[str] = _get_version()
