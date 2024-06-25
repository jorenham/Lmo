"""Typing compatibility for Python <3.13."""
import sys


if sys.version_info >= (3, 13):
    from typing import (
        LiteralString,
        ParamSpec,
        Protocol,
        Self,
        TypeVar,
        TypeVarTuple,
        Unpack,
        assert_never,
        assert_type,
        overload,
        runtime_checkable,
    )
else:
    from typing_extensions import (
        LiteralString,
        ParamSpec,
        Protocol,
        Self,
        TypeVar,
        TypeVarTuple,
        Unpack,
        assert_never,
        assert_type,
        overload,
        runtime_checkable,
    )


__all__ = (
    'LiteralString',
    'ParamSpec',
    'Protocol',
    'Self',
    'TypeVar',
    'TypeVarTuple',
    'Unpack',
    'assert_never',
    'assert_type',
    'overload',
    'runtime_checkable',
)
