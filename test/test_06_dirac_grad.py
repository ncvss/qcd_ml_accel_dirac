import torch
import pytest

import qcd_ml_accel_dirac
import qcd_ml

try:
    import gpt as g # type: ignore
    import qcd_ml_accel_dirac.compat

    # try to call the dirac wilson to see if the c++ function was compiled
    U = torch.zeros([4,2,2,2,2,3,3], dtype=torch.cdouble)
    v = torch.zeros([2,2,2,2,4,3], dtype=torch.cdouble)
    dw_avx = qcd_ml_accel_dirac.dirac_wilson_avx(U,-0.5)
    dwv_avx = dw_avx(v)

    # random gauge field used in test
    lat_dim = [8,8,8,16]
    rng = g.random("test06")
    U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
    grid = U_g[0].grid
    U = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(U_g))
    mass = -0.5
    csw = 1.0

    def test_wilson_avx_gradient_random():
        vref = torch.randn(lat_dim+[4,3], dtype=torch.cdouble, requires_grad=True)
        v = vref.clone().detach().requires_grad_(True)

        wref = qcd_ml.qcd.dirac.dirac_wilson(U, mass)
        resref = wref(vref)
        lossref = (resref * resref.conj()).real.sum()
        lossref.backward()

        w = qcd_ml_accel_dirac.dirac_wilson_avx(U, mass)
        res = w(v)
        loss = (res * res.conj()).real.sum()
        loss.backward()

        assert(torch.allclose(v.grad, vref.grad))


    def test_wilson_clover_avx_gradient_random():
        vref = torch.randn(lat_dim+[4,3], dtype=torch.cdouble, requires_grad=True)
        v = vref.clone().detach().requires_grad_(True)

        wref = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, csw)
        resref = wref(vref)
        lossref = (resref * resref.conj()).real.sum()
        lossref.backward()

        w = qcd_ml_accel_dirac.dirac_wilson_clover_avx(U, mass, csw)
        res = w(v)
        loss = (res * res.conj()).real.sum()
        loss.backward()

        assert(torch.allclose(v.grad, vref.grad))


except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_wilson_avx_gradient_random():
        pass

    @pytest.mark.skip("missing gpt")
    def test_wilson_clover_avx_gradient_random():
        pass

except RuntimeError:

    @pytest.mark.skip("missing AVX")
    def test_wilson_avx_gradient_random():
        pass

    @pytest.mark.skip("missing AVX")
    def test_wilson_clover_avx_gradient_random():
        pass
