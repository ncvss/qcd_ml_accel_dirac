import pytest
import numpy as np
import torch
import qcd_ml_accel_dirac

try:
    import gpt as g
    from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice


    def test_domain_wall():
        pass


except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_domain_wall():
        pass
