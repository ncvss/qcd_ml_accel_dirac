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
        m = 0.08
        m5 = 1.8
        
        mobius_params = {
            "mass": m,
            "M5": m5,
            "b": 0.0, #1.5?
            "c": 0.0, #0.5?
            "Ls": 8,
            "boundary_phases": [1.0, 1.0, 1.0, 1.0],
        }

        qm = g.qcd.fermion.mobius(U, mobius_params)
        grid5 = qm.F_grid
        src = g.vspincolor(grid5)
        rng.cnormal(src)

        res = qm * src
        res_torch = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(res))

        src_torch = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(src))
        U_torch = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(U))
        got = torch.ops.qcd_ml_accel_dirac.domain_wall_call(U_torch,src_torch,m,m5)

        assert torch.allclose(res_torch,got)


except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_domain_wall():
        pass
