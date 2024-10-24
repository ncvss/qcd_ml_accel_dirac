import torch
import qcd_ml_accel_dirac

# test how slow a call to C++ from Python is
def muladd_bench_nopar_py(a, b, c):
    return torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar(a, b, c)

def muladd_bench_par_py(a, b, c):
    return torch.ops.qcd_ml_accel_dirac.muladd_bench_par(a, b, c)
