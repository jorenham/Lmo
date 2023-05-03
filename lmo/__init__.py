from typing import Final as _Final

from ._meta import get_version as _get_version

from .univariate import *  # noqa
from .multivariate import *  # noqa


__version__: _Final[str] = _get_version()

