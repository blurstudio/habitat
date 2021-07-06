from habitat import Resolver
import os
import pytest


@pytest.fixture
def config_root():
    return os.path.dirname(__file__)


@pytest.fixture
def resolver(config_root):
    """Return a standard testing resolver"""
    config_paths = (os.path.join(config_root, "configs", "*"),)
    return Resolver(config_paths=config_paths)