import TyGrit


def test_import_package():
    """Test that the package can be imported."""
    assert TyGrit is not None


def test_math_sanity():
    """Basic sanity check."""
    assert 1 + 1 == 2
