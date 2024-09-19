import torch

import qcd_ml_accel_dirac


def test_shift_gaugemul_gauge(config_1500):
    shifted_config = torch.roll(config_1500[2], shifts = 1, dims = 1)

    expect = torch.matmul(config_1500[0], shifted_config)
    got = torch.ops.qcd_ml_accel_dirac.shift_gaugemul(config_1500[0], config_1500[2], [0,0,0,0], [0,-1,0,0])

    assert torch.allclose(expect, got)


def test_shift_gaugemul_vec(config_1500, psi_test):
    shifted_config = torch.roll(config_1500[1], shifts = 1, dims = 2)
    shifted_psi = torch.roll(psi_test, shifts = -1, dims = 0)

    expect = torch.matmul(shifted_psi, torch.transpose(shifted_config, -1, -2))
    got = torch.ops.qcd_ml_accel_dirac.shift_gaugemul(config_1500[1], psi_test, [0,0,-1,0], [1,0,0,0])

    assert torch.allclose(expect, got)