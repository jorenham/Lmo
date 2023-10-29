"""Integrations and extensions for 3rd party packages."""
import contextlib

__all__ = ('install',)


def install():
    """
    Install the extensions for all available 3rd party packages.

    There should be no need to call this manually: this is done automatically
    when `lmo` is imported.
    """
    from . import scipy_stats
    scipy_stats.install()

    with contextlib.suppress(ImportError):
        from . import pandas
        pandas.install()
