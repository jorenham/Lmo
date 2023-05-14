__all__ = 'expand_trimming',

from lmo.typing import Trimming


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
