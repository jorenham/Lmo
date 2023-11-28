"""Mathematical constants."""

__all__ = (
    'theta_m',
    'theta_m_bar',
)

from typing import Final

theta_m: Final[float] = 0.955316618124509278163857102515757754
r"""
Magic angle \( \theta_m = \arctan \sqrt 2 \approx 0.9553 \).

See also:
    - https://wikipedia.org/wiki/Magic_angle
"""

theta_m_bar: Final[float] = 0.221389709640985827437970269154380360
r"""
Magic number of turns \( \bar{\theta}_m = \theta_m / (2 \pi) \approx 0.2214 \)

See also:
    - [theta_m][lmo.constants.theta_m]
    - https://wikipedia.org/wiki/Magic_angle
    - https://wikipedia.org/wiki/Turn_(angle)
"""
