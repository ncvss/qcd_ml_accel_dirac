import torch
import torch.utils.benchmark as benchmark
import socket
import numpy as np
import time
import pytest

import qcd_ml_accel_dirac
import qcd_ml


@pytest.mark.benchmark
def test_perf_counter_wilson():
    print()
    num_threads = torch.get_num_threads()
    print("running on host", socket.gethostname())
    print(f"Machine has {num_threads} threads")

    n_measurements = 500
    n_warmup = 20

    lat_dim = [16,16,16,32]
    print("lattice dimensions:",lat_dim)

    U = torch.randn([4]+lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)
    mass = -0.5
    print("mass parameter:",mass)

    dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
    dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)

    dwv_py = dw_py(v)
    dwv_cpp = dw_cpp(v)

    for _ in range(n_warmup):
        dwv_py = dw_py(v)
        dwv_cpp = dw_cpp(v)


    results_py = np.zeros(n_measurements)
    results_cpp = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        dwv_py = dw_py(v)
        stop = time.perf_counter_ns()
        results_py[i] = stop - start

        start = time.perf_counter_ns()
        dwv_cpp = dw_cpp(v)
        stop = time.perf_counter_ns()
        results_cpp[i] = stop - start

        start = time.perf_counter_ns()
        stop = time.perf_counter_ns()
        bias[i] = stop - start


    results_py_sorted = np.sort(results_py)[:(n_measurements // 5)]
    results_cpp_sorted = np.sort(results_cpp)[:(n_measurements // 5)]


    for lang_name,results_sorted in [["qcd_ml",results_py_sorted], ["qcd_ml_accel_dirac",results_cpp_sorted]]:
        print("-------")
        print(lang_name)
        print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
        print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
        print(f"best : [us] {results_sorted[0]/1000}")
        print(f"mean bias : [us] {np.mean(bias)/1000}")
        print(f"std bias : [us] {np.mean(bias)/1000}")

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
        

    assert torch.allclose(dwv_cpp,dwv_py)

@pytest.mark.benchmark
def test_perf_counter_wilson_clover():
    print()
    num_threads = torch.get_num_threads()
    print("running on host", socket.gethostname())
    print(f"Machine has {num_threads} threads")
    
    n_measurements = 500
    n_warmup = 20

    lat_dim = [16,16,16,32]
    print("lattice dimensions:",lat_dim)

    U = torch.randn([4]+lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)
    mass = -0.5
    csw = 1.0
    print("mass parameter:",mass)
    print("c_sw:",csw)
    

    dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
    dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)

    dwcv_py = dwc_py(v)
    dwcv_cpp = dwc_cpp(v)

    for _ in range(n_warmup):
        dwcv_py = dwc_py(v)
        dwcv_cpp = dwc_cpp(v)


    results_py = np.zeros(n_measurements)
    results_cpp = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        dwcv_py = dwc_py(v)
        stop = time.perf_counter_ns()
        results_py[i] = stop - start

        start = time.perf_counter_ns()
        dwcv_cpp = dwc_cpp(v)
        stop = time.perf_counter_ns()
        results_cpp[i] = stop - start

        start = time.perf_counter_ns()
        stop = time.perf_counter_ns()
        bias[i] = stop - start


    results_py_sorted = np.sort(results_py)[:(n_measurements // 5)]
    results_cpp_sorted = np.sort(results_cpp)[:(n_measurements // 5)]


    for lang_name,results_sorted in [["qcd_ml",results_py_sorted], ["qcd_ml_accel_dirac",results_cpp_sorted]]:
        print("-------")
        print(lang_name)
        print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
        print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
        print(f"best : [us] {results_sorted[0]/1000}")
        print(f"mean bias : [us] {np.mean(bias)/1000}")
        print(f"std bias : [us] {np.mean(bias)/1000}")

        data_size = v.element_size() * v.nelement() * 2 + U.element_size() * U.nelement() * 10/4
        data_size_MiB = data_size / 1024**2

        print()
        print(f"data : [MiB] {data_size_MiB: .3f}")

        throughput = data_size / (np.mean(results_sorted) / 1000**3)
        throughput_GiBs = throughput / 1024 ** 3
        throughput_peak = data_size / (results_sorted[0] / 1000**3)
        throughput_peak_GiBs = throughput_peak / 1024 ** 3

        print(f"throughput : [GiB/s] {throughput_GiBs: .3f}")
        print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")
        

    assert torch.allclose(dwcv_cpp,dwcv_py)
