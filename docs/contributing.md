# Contributing to *Lmo*

Any contributions to *Lmo* are appreciated!

## Issues

Questions, feature requests and bug reports are all welcome as issues.

When reporting a bug, make sure to include the versions of `lmo`, `python`,
`numpy` and `scipy` you are using, and provide a **reproducible** example of
the bug.

## Environment setup

Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/)
installed.
You can install the dev dependencies using

```bash
uv sync
```

### Testing

Lmo uses [pytest](https://docs.pytest.org/en/stable/) and
[hypothesis](https://hypothesis.readthedocs.io/en/latest/) as testing
framework.

The tests can be run using

```bash
uv run pytest
```

## Documentation

If your change involves documentation updates, you can conjure up a live
preview:

```bash
uv run mkdocs serve
```

This will require `pandoc` and `pandoc-citeproc` to be installed on your
system (e.g. using the conda-forge `pypandoc` package, on
`apt-get install pandoc pandoc-citeproc` on Ubuntu).

This will make the site available at `http://127.0.0.1:8000/`.
It automatically reloads when changes are made to the source code or the
documentation.

But don't worry about building the docs, or bumping the version;
Lmo's personal assistant will do that on release.
