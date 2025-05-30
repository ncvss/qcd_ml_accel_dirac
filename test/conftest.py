import pytest
import torch
import numpy as np

import os


def pytest_addoption(parser):
    parser.addoption(
        "--runbenchmark", action="store_true", default=False, help="run benchmark tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as benchmark tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runbenchmark"):
        # --runbenchmark given in cli: do not skip benchmark tests
        return
    skip_bench = pytest.mark.skip(reason="need --runbenchmark option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_bench)


@pytest.fixture 
def config_1500():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","1500.config.npy")))

@pytest.fixture 
def psi_test():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","psi_test.npy")))

@pytest.fixture 
def psi_Dw1500_m0p5_psitest():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","psi_Dw1500_m0p5_psitest.npy")))

@pytest.fixture 
def psi_Dwc1500_m0p5_psitest():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","psi_Dwc1500_m0p5_psitest.npy")))

@pytest.fixture 
def config_1500_gtrans_1200mu0():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","1500_gtrans_1200mu0.npy")))

