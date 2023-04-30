__all__ = (
    '__version__',
    'l',
)

import importlib.metadata
from typing import Final

__version__: Final[str] = importlib.metadata.version(
    __package__ or __file__.split('/')[-1]
)


from . import l
