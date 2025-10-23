"""Test module for tester package."""

import pytest


def test_import_main():
    """Test importing and running the main function."""
    from tester.main import main

    # Check that everything is working, and catch Dora RuntimeError
    # as we're not running in a Dora dataflow.
    with pytest.raises(RuntimeError):
        main()
