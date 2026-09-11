import pytest


def pytest_itemcollected(item):
    item.add_marker(pytest.mark.gw)
