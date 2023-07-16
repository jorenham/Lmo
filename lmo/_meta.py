__all__ = ('get_version',)

import importlib.metadata


def get_version() -> str:
    return importlib.metadata.version(__package__ or __file__.split('/')[-1])
