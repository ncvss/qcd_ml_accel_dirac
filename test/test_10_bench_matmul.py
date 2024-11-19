import torch
import socket
import time
import numpy as np

import qcd_ml_accel_dirac

def test_throughput_big_matmul():
    print()
    print("running on host", socket.gethostname())

    U_dim = [8*16*3,8*16*3]
    v_dim = [8*16*3,8*16*4]
    print("matrix sizes:",U_dim,v_dim)

    U = torch.randn(U_dim,dtype=torch.cdouble)
    v = torch.randn(v_dim,dtype=torch.cdouble)

    n_measurements = 200
    n_warmup = 20

    vp = torch.ops.qcd_ml_accel_dirac.big_matmul(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.big_matmul(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.big_matmul(U,v)
        stop = time.perf_counter_ns()
        results[i] = stop - start

        start = time.perf_counter_ns()
        stop = time.perf_counter_ns()
        bias[i] = stop - start

    results_sorted = np.sort(results)[:(n_measurements // 5)]

    print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
    print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
    print(f"best : [us] {results_sorted[0]/1000}")
    print(f"mean bias : [us] {np.mean(bias)/1000}")
    print(f"std bias : [us] {np.mean(bias)/1000}")

    data_size = 2 * v.element_size() * v.nelement() + U.element_size() * U.nelement()
    data_size_MiB = data_size / 1024**2

    print()
    print(f"data : [MiB] {data_size_MiB: .3f}")

    throughput = data_size / (np.mean(results_sorted) / 1000**3)
    throughput_GiBs = throughput / 1024 ** 3
    throughput_peak = data_size / (results_sorted[0] / 1000**3)
    throughput_peak_GiBs = throughput_peak / 1024 ** 3

    print(f"throughput : [GiB/s] {throughput_GiBs: .3f}")
    print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")
    
    assert torch.allclose(vp, torch.matmul(U,v))

