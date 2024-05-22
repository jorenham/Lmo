# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
import contextlib
from collections.abc import Iterator

import pytest

from lmo import _lm


@contextlib.contextmanager
def tmp_cache() -> Iterator[_lm._Cache]:
    cache_tmp: _lm._Cache = {}
    cache_old, _lm._CACHE = _lm._CACHE, cache_tmp
    try:
        yield cache_tmp
    finally:
        _lm._CACHE = cache_old


@pytest.fixture(name='tmp_cache')
def tmp_cache_fixture():
    with tmp_cache() as cache:
        assert not cache
        yield cache
