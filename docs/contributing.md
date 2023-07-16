# Contributing to *Lmo*

Any contributions to *Lmo* are appreciated!

## Issues

Questions, feature requests and bug reports are all welcome as issues.

When reporting a bug, make sure to include the versions of `lmo`, `python`, 
`numpy` and `scipy` you are using, and provide a **reproducible** example of 
the bug.

## Development

Ensure you have [poetry](https://python-poetry.org/docs/#installation) 
installed, then clone your fork, and install with

```bash
poetry install --sync
```

It can help to use Lmo's lowest-supported Python version, so that you don't
accidentally use those bleeding-edge Python features that you shouldn't, 
`poetry env use python3.x`

Now you can go ahead and do your thing. 
And don't forget the type annotations, add tests, and to lint it all. 

If you're a 10x developer that doesn't wait on CI workflows, you can use the 
following 1337 shellscript (keep in mind that the CI runs this on all supported
Python versions):

```bash
poetry run ruff check lmo
poetry run pyright
poetry run py.test
```

If your change involves documentation updates, you can conjure up a live 
preview:

```bash
poetry run mkdocs serve
```

But don't worry about building the docs, or bumping the version;
Lmo's personal assistant will do that on release.
