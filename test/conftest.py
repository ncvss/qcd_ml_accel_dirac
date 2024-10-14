import pytest
import torch
import numpy as np

import os



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

