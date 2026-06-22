def test_import():
    import cellpose_tools
    assert hasattr(cellpose_tools, "__version__")