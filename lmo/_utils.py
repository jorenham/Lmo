__all__ = 'clean_order', 'ensure_axis_at', 'as_float_array'

from typing import Any, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from lmo.typing import IndexOrder

T = TypeVar('T', bound=np.generic)

def clean_order(
    order: SupportsIndex,
    /,
    name: str = 'r',
    strict: bool = False,
) -> int:
    if (r := order.__index__()) < (r0 := int(strict)):
        raise TypeError(f'expected {name} >= {r0}, got {r}')

    return r


def ensure_axis_at(
    a: npt.NDArray[T],
    /,
    source: int | None,
    destination: int,
    order: IndexOrder = 'C',
) -> npt.NDArray[T]:
    if a.ndim <= 1 or source == destination:
        return a
    if source is None:
        return a.ravel(order)

    source = source + a.ndim if source < 0 else source
    destination = destination + a.ndim if destination < 0 else destination

    return a if source == destination else np.moveaxis(a, source, destination)


def as_float_array(
    a: npt.ArrayLike,
    /,
    dtype: npt.DTypeLike = None,
    order: IndexOrder | None = None,
    *,
    check_finite: bool = False,
    flat: bool = False
) -> npt.NDArray[np.floating[Any]]:
    """
    Convert to array if needed, and only cast to float64 dtype if not a
    floating type already. Similar as in e.g. `numpy.mean`.
    """
    asarray = np.asarray_chkfinite if check_finite else np.asarray

    x = asarray(a, dtype=dtype, order=order)
    out = x if isinstance(x.dtype.type, np.floating) else x.astype(np.float_)

    # the `_[()]` ensures that 0-d arrays become scalars
    return (out.ravel() if flat and out.ndim != 1 else out)[()]
