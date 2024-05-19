"""Typing compatibility for Python <3.11."""

import sys


if sys.version_info < (3, 11):
    from typing_extensions import (
        LiteralString,
        Self,
        TypeVarTuple,
        Unpack,
        assert_type,
    )
else:
    from typing import LiteralString, Self, TypeVarTuple, Unpack, assert_type


__all__ = 'LiteralString', 'Self', 'TypeVarTuple', 'Unpack', 'assert_type'
