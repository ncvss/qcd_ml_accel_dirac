import torch
import qcd_ml_accel_dirac


def test_plaquette_both_langs(config_1500_gtrans_1200mu0):
    plaqpy = qcd_ml_accel_dirac.plaq_action_py(config_1500_gtrans_1200mu0,2.0)
    plaqcpp = torch.ops.qcd_ml_accel_dirac.plaquette_action(config_1500_gtrans_1200mu0,2.0)
    assert abs(plaqpy-plaqcpp) < 0.0001 * abs(plaqpy)

def test_plaquette_py_gauge_invariance(config_1500,config_1500_gtrans_1200mu0):

    plaqpy = qcd_ml_accel_dirac.plaq_action_py(config_1500,2.0)
    plaqpy_trans = qcd_ml_accel_dirac.plaq_action_py(config_1500_gtrans_1200mu0,2.0)
    assert abs(plaqpy-plaqpy_trans) < 0.0001 * abs(plaqpy)

def test_plaquette_cpp_gauge_invariance(config_1500,config_1500_gtrans_1200mu0):

    plaqcpp = torch.ops.qcd_ml_accel_dirac.plaquette_action(config_1500,2.0)
    plaqcpp_trans = torch.ops.qcd_ml_accel_dirac.plaquette_action(config_1500_gtrans_1200mu0,2.0)
    assert abs(plaqcpp-plaqcpp_trans) < 0.0001 * abs(plaqcpp)

# all tests pass, so gauge invariance is preserved
# however, the result is far too large to be correct (should be around 0.6)
# Gattringer (3.5) and Fig. 3.5 suggest g is around 1
# According to gpt/qcd/gauge/stencil/plaquette, we need to normalize this with the number
# of different plaquettes and the volume, which yields around 0.6
