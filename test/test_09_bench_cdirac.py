import torch
import numpy as np
import time

import qcd_ml_accel_dirac


def test_wilson_pure_c():
    print()
    n_measurements = 1000
    n_warmup = 10

    lat_dim = [8,8,8,16]
    print("lattice dimensions:",lat_dim)

    U = torch.randn([4]+lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)
    dummy = torch.tensor([1],dtype=torch.cdouble)
    mass = -0.5
    print("mass parameter:",mass)

    ti = torch.ops.qcd_ml_accel_dirac.dw_call_c_test(dummy)

    for _ in range(n_warmup):
        ti = torch.ops.qcd_ml_accel_dirac.dw_call_c_test(dummy)


    results = np.zeros(n_measurements)

    for i in range(n_measurements):
        results[i] = torch.ops.qcd_ml_accel_dirac.dw_call_c_test(dummy)


    results_sorted = np.sort(results)[:(n_measurements // 5)]

    print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
    print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
    print(f"best : [us] {results_sorted[0]/1000}")

    data_size = v.element_size() * v.nelement() * 2 + U.element_size() * U.nelement()
    data_size_MiB = data_size / 1024**2

    print()
    print(f"data : [MiB] {data_size_MiB: .3f}")

    throughput = data_size / (np.mean(results_sorted) / 1000**3)
    throughput_GiBs = throughput / 1024 ** 3
    throughput_peak = data_size / (results_sorted[0] / 1000**3)
    throughput_peak_GiBs = throughput_peak / 1024 ** 3

    print(f"throughput : [GiB/s] {throughput_GiBs: .3f}")
    print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")
        

    assert True
