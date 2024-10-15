# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from lmo import _lm

if TYPE_CHECKING:
    from collections.abc import Generator


@contextlib.contextmanager
def tmp_cache() -> Generator[_lm._Cache, None, None]:
    cache_tmp: _lm._Cache = {}
    cache_old, _lm._CACHE = _lm._CACHE, cache_tmp
    try:
        yield cache_tmp
    finally:
        _lm._CACHE = cache_old


@pytest.fixture(name="tmp_cache")
def tmp_cache_fixture() -> Generator[_lm._Cache, None, None]:
    with tmp_cache() as cache:
        assert not cache
        yield cache
