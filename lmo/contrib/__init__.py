"""Integrations and extensions for 3rd party packages."""

try:
    import pandas as pd
except ImportError:
    pd = None

__all__ = ("install",)


def install() -> None:
    """
    Install the extensions for all available 3rd party packages.

    There should be no need to call this manually: this is done automatically
    when `lmo` is imported.
    """
    from .scipy_stats import install as install_scipy_stats

    install_scipy_stats()

    if pd is not None:
        from .pandas import install as install_pandas

        install_pandas()
