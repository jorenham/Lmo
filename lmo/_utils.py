__all__ = 'clean_order', 'expand_trimming'

from typing import SupportsIndex

from lmo.typing import Trimming


def clean_order(
    order: SupportsIndex,
    /,
    name: str = 'r',
    strict: bool = False,
) -> int:
    if (r := order.__index__()) < (r0 := int(strict)):
        raise TypeError(f'expected {name} >= {r0}, got {r}')

    return r


def expand_trimming(trim: Trimming, /) -> tuple[int, int]:
    match trim:
        case () | False:
            return 0, 0
        case (int(t),) | int(t) if t >= 0:
            return t, t
        case (int(tl), int(tr)) if min(tl, tr) >= 0:
            return tl, tr
        case _:
            raise TypeError(f'{trim!r} is not a valid trimming')


