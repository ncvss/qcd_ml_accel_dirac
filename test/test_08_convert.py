import torch
import numpy as np
import pytest

from qcd_ml_accel_dirac import lattice_to_array, array_to_lattice, array_to_lattice_list

try:
    import gpt as g # type: ignore

    def test_conversion_gpt_torch_gpt():
        rng = g.random("convert")
        U = g.qcd.gauge.random(g.grid([8,8,8,16], g.double), rng)
        grid = U[0].grid
        v = rng.cnormal(g.vspincolor(grid))

        U_conv = array_to_lattice_list(lattice_to_array(U), grid, g.mcolor)
        v_conv = array_to_lattice(lattice_to_array(v), grid, g.vspincolor)
        
        assert all([g.norm2(U[i] - U_conv[i]) / g.norm2(U[i]) < 1e-14 for i in range(len(U))]
                   + [g.norm2(v - v_conv) / g.norm2(v) < 1e-14])


    def test_conversion_torch_gpt_torch():
        U = torch.randn([4,8,8,8,16,3,3], dtype=torch.cdouble)
        v = torch.randn([8,8,8,16,4,3], dtype=torch.cdouble)

        grid = g.grid([8,8,8,16], g.double)

        U_conv = torch.tensor(lattice_to_array(array_to_lattice_list(U, grid, g.mcolor)))
        v_conv = torch.tensor(lattice_to_array(array_to_lattice(v, grid, g.vspincolor)))

        assert all([torch.allclose(U, U_conv), torch.allclose(v, v_conv)])


except ImportError:
    @pytest.mark.skip("missing gpt")
    def test_conversion_gpt_torch_gpt():
        pass

    @pytest.mark.skip("missing gpt")
    def test_conversion_torch_gpt_torch():
        pass
