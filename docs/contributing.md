# Contributing to *Lmo*

Any contributions to *Lmo* are appreciated!

## Issues

Questions, feature requests and bug reports are all welcome as issues.

When reporting a bug, make sure to include the versions of `lmo`, `python`,
`numpy` and `scipy` you are using, and provide a **reproducible** example of
the bug.

## Environment setup

Ensure you have [poetry](https://python-poetry.org/docs/#installation)
installed.
It can help to use Lmo's lowest-supported Python version, so that you don't
accidentally use those bleeding-edge Python features that you shouldn't, e.g.

```bash
poetry env use python3.10
```

Now you can install the dev dependencies using

```bash
poetry install --sync
```

### pre-commit

Lmo uses [pre-commit](https://pre-commit.com/) to ensure that the code is
formatted and typed correctly when committing the changes.

```bash
poetry run pre-commit install
```

It can also be manually run:

```bash
poetry run pre-commit --all-files
```

### Testing

Lmo uses [pytest](https://docs.pytest.org/en/stable/) and
[hypothesis](https://hypothesis.readthedocs.io/en/latest/) as testing
framework.

The tests can be run using

```bash
poetry run pytest
```

## Documentation

If your change involves documentation updates, you can conjure up a live
preview:

```bash
poetry run mkdocs serve
```

This will make the site available at `http://127.0.0.1:8000/`.
It automatically reloads when changes are made to the source code or the
documentation.

But don't worry about building the docs, or bumping the version;
Lmo's personal assistant will do that on release.
