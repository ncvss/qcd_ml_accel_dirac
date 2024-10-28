import torch 

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

