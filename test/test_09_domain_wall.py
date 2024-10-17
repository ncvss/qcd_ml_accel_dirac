import pytest
import numpy as np
import torch
import qcd_ml_accel_dirac

try:
    import gpt as g
    from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice


    def test_domain_wall():
        rng = g.random("test")
        U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng, scale=2.0)
        grid = U[0].grid

        mobius_params = {
            "mass": 0.08,
            "M5": 1.8,
            "b": 1.5,
            "c": 0.5,
            "Ls": 8,
            "boundary_phases": [1.0, 1.0, 1.0, 1.0],
        }

        qm = g.qcd.fermion.mobius(U, mobius_params)


except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_domain_wall():
        pass
