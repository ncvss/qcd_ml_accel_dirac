import torch
import pytest

import qcd_ml_accel_dirac
import qcd_ml


def test_dirac_wilson_precomputed(config_1500, psi_test, psi_Dw1500_m0p5_psitest):
    w = qcd_ml_accel_dirac.dirac_wilson(config_1500, -0.5)

    expect = psi_Dw1500_m0p5_psitest
    got = w(psi_test)
    assert torch.allclose(expect, got)


def test_dirac_wilson_clover_precomputed(config_1500, psi_test, psi_Dwc1500_m0p5_psitest):
    w = qcd_ml_accel_dirac.dirac_wilson_clover(config_1500, -0.5, 1)

    expect = psi_Dwc1500_m0p5_psitest
    got = w(psi_test)
    assert torch.allclose(expect, got)


def test_dirac_wilson_random():

    lat_dim = [8,8,8,16]

    # not a gauge field, but can still be used for testing equality
    U = torch.randn([4]+lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)
    mass = -0.5

    dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
    dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)

    assert torch.allclose(dw_cpp(v),dw_py(v))


def test_dirac_wilson_clover_random():

    lat_dim = [8,8,8,16]

    # not a gauge field, but can still be used for testing equality
    U = torch.randn([4]+lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)
    mass = -0.5
    csw = 1.0

    dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
    dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)

    assert torch.allclose(dwc_cpp(v),dwc_py(v))

try:
    # try to call the dirac wilson to see if the c++ function was compiled
    U = torch.zeros([4,2,2,2,2,3,3], dtype=torch.cdouble)
    v = torch.zeros([2,2,2,2,4,3], dtype=torch.cdouble)
    dw_avx = qcd_ml_accel_dirac.dirac_wilson_avx(U,-0.5)
    dwv_avx = dw_avx(v)

except RuntimeError:

    @pytest.mark.skip("missing AVX")
    def test_dirac_wilson_avx_precomputed():
        pass

    @pytest.mark.skip("missing AVX")
    def test_dirac_wilson_clover_avx_old_precomputed():
        pass

    @pytest.mark.skip("missing AVX")
    def test_dirac_wilson_clover_avx_precomputed():
        pass

    @pytest.mark.skip("missing AVX")
    def test_dirac_wilson_avx_random():
        pass

    @pytest.mark.skip("missing AVX")
    def test_dirac_wilson_clover_avx_old_random():
        pass

    @pytest.mark.skip("missing AVX")
    def test_dirac_wilson_clover_avx_random():
        pass

else:

    def test_dirac_wilson_avx_precomputed(config_1500, psi_test, psi_Dw1500_m0p5_psitest):
        w = qcd_ml_accel_dirac.dirac_wilson_avx(config_1500, -0.5)

        expect = psi_Dw1500_m0p5_psitest
        got = w(psi_test)
        assert torch.allclose(expect, got)


    def test_dirac_wilson_clover_avx_old_precomputed(config_1500, psi_test, psi_Dwc1500_m0p5_psitest):
        w = qcd_ml_accel_dirac.dirac_wilson_clover_avx_old(config_1500, -0.5, 1)

        expect = psi_Dwc1500_m0p5_psitest
        got = w(psi_test)
        assert torch.allclose(expect, got)


    def test_dirac_wilson_clover_avx_precomputed(config_1500, psi_test, psi_Dwc1500_m0p5_psitest):
        w = qcd_ml_accel_dirac.dirac_wilson_clover_avx(config_1500, -0.5, 1)

        expect = psi_Dwc1500_m0p5_psitest
        got = w(psi_test)
        assert torch.allclose(expect, got)


    try:
        import gpt as g
        import qcd_ml_accel_dirac.compat

        lat_dim = [8,8,8,16]

        rng = g.random("test01")
        U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_g[0].grid
        v_g = rng.cnormal(g.vspincolor(grid))

        U_rand = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(U_g))
        v_rand = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(v_g))

        mass = -0.5
        csw = 1.0


        def test_dirac_wilson_avx_random():
            w = qcd_ml_accel_dirac.dirac_wilson_avx(U_rand, mass)
            wref = qcd_ml.qcd.dirac.dirac_wilson(U_rand, mass)

            expect = wref(v_rand)
            got = w(v_rand)

            assert torch.allclose(expect, got)


        def test_dirac_wilson_clover_avx_old_random():
            w = qcd_ml_accel_dirac.dirac_wilson_clover_avx_old(U_rand, mass, csw)
            wref = qcd_ml.qcd.dirac.dirac_wilson_clover(U_rand, mass, csw)

            expect = wref(v_rand)
            got = w(v_rand)
            assert torch.allclose(expect, got)


        def test_dirac_wilson_clover_avx_random():
            w = qcd_ml_accel_dirac.dirac_wilson_clover_avx(U_rand, mass, csw)
            wref = qcd_ml.qcd.dirac.dirac_wilson_clover(U_rand, mass, csw)

            expect = wref(v_rand)
            got = w(v_rand)
            assert torch.allclose(expect, got)
    

    except ImportError:

        @pytest.mark.skip("missing GPT")
        def test_dirac_wilson_avx_random():
            pass

        @pytest.mark.skip("missing GPT")
        def test_dirac_wilson_clover_avx_old_random():
            pass

        @pytest.mark.skip("missing GPT")
        def test_dirac_wilson_clover_avx_random():
            pass

    