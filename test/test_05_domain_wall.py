import pytest
import numpy as np
import torch
import qcd_ml_accel_dirac


try:
    import gpt as g # type: ignore
    import qcd_ml_accel_dirac.compat
except ImportError:
    gpt_missing = True
else:
    gpt_missing = False


@pytest.mark.skipif(gpt_missing, reason="missing GPT")
def test_domain_wall():
    print()
    rng = g.random("test05")
    U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)
    grid = U[0].grid
    m = 0.08
    m5 = 1.8
    
    mobius_params = {
        "mass": m,
        "M5": m5,
        "b": 1.0,
        "c": 0.0,
        "Ls": 8,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    }

    qm = g.qcd.fermion.mobius(U, mobius_params)
    grid5 = qm.F_grid
    src = g.vspincolor(grid5)
    rng.cnormal(src)

    res = qm(src)

    res_torch = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(res))

    src_torch = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(src))
    U_torch = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(U))

    qm_cpp = qcd_ml_accel_dirac.domain_wall_dirac(U_torch,m,m5)
    got = qm_cpp(src_torch)

    assert torch.allclose(res_torch,got)
