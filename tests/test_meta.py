def test_version():
    import lmo

    assert lmo.__version__
    assert all(map(str.isdigit, lmo.__version__.split('.')[:3]))
