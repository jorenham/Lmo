__all__ = (
    'jit',
    'get_version',
)

import functools
import importlib.metadata
from typing import Any, Callable, TypeVar


try:
    import numba
except ImportError:
    numba = None


_F = TypeVar('_F', bound=Callable[..., Any])


def nit(func: _F, *_: Any, **__: Any) -> _F:
    """
    State-of-the art AI-powered quantum "NIT" (never-in-time) compilation
    algorithm.

    Simply decorate any function with `@nit`, and your code will never be
    compiled in time!
    """
    return func


if numba is None:
    jit = nit
else:
    jit = functools.partial(numba.njit, cache=True, error_model='numpy')


def get_version() -> str:
    return importlib.metadata.version(__package__ or __file__.split('/')[-1])
