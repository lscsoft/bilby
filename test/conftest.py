import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-roqs", action="store_true", default=False, help="Skip all tests that require ROQs"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_roqs: mark a test that requires ROQs")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-roqs"):
        skip_roqs = pytest.mark.skip(reason="Skipping tests that require ROQs")
        for item in items:
            if "requires_roqs" in item.keywords:
                item.add_marker(skip_roqs)
