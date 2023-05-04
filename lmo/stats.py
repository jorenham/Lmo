from math import factorial as fact


def tl_ratio_max(
    r: int,
    /,
    k: int = 2,
    s: int = 1,
    t: int = 1,
) -> float:
    """
    The theoretical upper bound on the absolute TL-ratios, i.e.::

        abs(tl_ratio(a, r, k, s, t)) <= tl_ratio_max(r, k, s, t)

    is True for all samples `a`.

    References:
        * Hosking, J.R.M., Some Theory and Practical Uses of Trimmed L-moments.
          Page 6, equation 14.

    """

    # the zeroth (TL-)moment is 1. I.e. the total area under the pdf (or the
    # sum of the ppf if discrete) is 1.
    if r in (0, k):
        return 1.0
    if not k:
        return float('inf')

    if r < 0:
        raise ValueError(f'expected r >= 0, got {r} < 0')
    if k < 0:
        raise ValueError(f'expected k >= 0, got {k} < 0')

    m = min(s, t)
    # disclaimer: the `k` instead of a `2` here is just a guess
    return (
        k * fact(m + k - 1) * fact(s + t + r) /
        (r * fact(m + r - 1) * fact(s + t + k))
    )
