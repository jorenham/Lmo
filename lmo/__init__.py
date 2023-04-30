from typing import Final as _Final

from ._meta import get_version as _get_version

from .l_univariate import *  # noqa
from .l_multivariate import *  # noqa


__version__: _Final[str] = _get_version()

