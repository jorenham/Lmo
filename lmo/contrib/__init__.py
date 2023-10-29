"""Integrations and extensions for 3rd party packages."""

__all__ = ('install',)

from . import scipy_stats


def install():
    """
    Install the extensions for all available 3rd party packages.

    There should be no need to call this manually: this is done automatically
    when `lmo` is imported.
    """
    scipy_stats.install()
