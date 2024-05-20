"""Typing compatibility for Python <3.11."""
import sys


if sys.version_info < (3, 13):
    from typing_extensions import (
        LiteralString,
        ParamSpec,
        Self,
        TypeVar,
        TypeVarTuple,
        Unpack,
        assert_never,
        assert_type,
    )
else:
    from typing import (
        LiteralString,
        ParamSpec,
        Self,
        TypeVar,
        TypeVarTuple,
        Unpack,
        assert_never,
        assert_type,
    )


__all__ = (
    'LiteralString',
    'ParamSpec',
    'Self',
    'TypeVar',
    'TypeVarTuple',
    'Unpack',
    'assert_never',
    'assert_type',
)
