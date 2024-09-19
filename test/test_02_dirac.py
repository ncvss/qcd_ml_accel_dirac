import torch 

import qcd_ml_accel_dirac
#import qcd_ml_accel_dirac_cpp


def test_dirac_wilson_precomputed(config_1500, psi_test, psi_Dw1500_m0p5_psitest):
    w = qcd_ml_accel_dirac.ops.dirac_wilson_cpp(config_1500, -0.5)
    expect = psi_Dw1500_m0p5_psitest

    got = w(psi_test)

    assert torch.allclose(expect, got)


def test_dirac_wilson_clover_precomputed(config_1500, psi_test, psi_Dwc1500_m0p5_psitest):
    w = qcd_ml_accel_dirac.ops.dirac_wilson_clover_cpp(config_1500, -0.5, 1)
    expect = psi_Dwc1500_m0p5_psitest

    got = w(psi_test)

    assert torch.allclose(expect, got)
