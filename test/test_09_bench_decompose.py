import torch
import torch.utils.benchmark as benchmark
import socket
import time
import numpy as np

import qcd_ml_accel_dirac


# test the speed of different sub-computations of the dirac wilson operator

def test_throughput_matmul():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform(U,v)
    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_par(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform(U,v)
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_par(U,v)

    results_n = np.zeros(n_measurements)
    results_p = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform(U,v)
        stop = time.perf_counter_ns()
        results_n[i] = stop - start

        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_par(U,v)
        stop = time.perf_counter_ns()
        results_p[i] = stop - start

        start = time.perf_counter_ns()
        stop = time.perf_counter_ns()
        bias[i] = stop - start

    results_n_sorted = np.sort(results_n)[:(n_measurements // 5)]
    results_p_sorted = np.sort(results_p)[:(n_measurements // 5)]

    for ty,results_sorted in [["not parallel",results_n_sorted],["parallel",results_p_sorted]]:
        print("-----")
        print(ty)
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
    
    assert True


def test_throughput_matmul_diff_layout():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    vs = v.transpose(4,5).contiguous()

    n_measurements = 9000
    n_warmup = 500

    uv = torch.ops.qcd_ml_accel_dirac.gauge_transform_par(U,v)
    uvs = torch.ops.qcd_ml_accel_dirac.gauge_transform_spin_inner(U,vs)
    uvg = torch.ops.qcd_ml_accel_dirac.gauge_transform_gauge_inner(U,v)
    uvu = torch.ops.qcd_ml_accel_dirac.gauge_transform_gauge_unroll(U,v)

    for _ in range(n_warmup):
        uv = torch.ops.qcd_ml_accel_dirac.gauge_transform_par(U,v)
        uvs = torch.ops.qcd_ml_accel_dirac.gauge_transform_spin_inner(U,vs)
        uvg = torch.ops.qcd_ml_accel_dirac.gauge_transform_gauge_inner(U,v)
        uvu = torch.ops.qcd_ml_accel_dirac.gauge_transform_gauge_unroll(U,v)

    results_n = np.zeros(n_measurements)
    results_s = np.zeros(n_measurements)
    results_g = np.zeros(n_measurements)
    results_u = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        uv = torch.ops.qcd_ml_accel_dirac.gauge_transform_par(U,v)
        stop = time.perf_counter_ns()
        results_n[i] = stop - start

        start = time.perf_counter_ns()
        uvs = torch.ops.qcd_ml_accel_dirac.gauge_transform_spin_inner(U,vs)
        stop = time.perf_counter_ns()
        results_s[i] = stop - start

        start = time.perf_counter_ns()
        uvg = torch.ops.qcd_ml_accel_dirac.gauge_transform_gauge_inner(U,v)
        stop = time.perf_counter_ns()
        results_g[i] = stop - start

        start = time.perf_counter_ns()
        uvu = torch.ops.qcd_ml_accel_dirac.gauge_transform_gauge_unroll(U,v)
        stop = time.perf_counter_ns()
        results_u[i] = stop - start

        start = time.perf_counter_ns()
        stop = time.perf_counter_ns()
        bias[i] = stop - start

    results_n_sorted = np.sort(results_n)[:(n_measurements // 5)]
    results_s_sorted = np.sort(results_s)[:(n_measurements // 5)]
    results_g_sorted = np.sort(results_g)[:(n_measurements // 5)]
    results_u_sorted = np.sort(results_u)[:(n_measurements // 5)]

    for ty,results_sorted in [["gauge innermost in memory, spin innermost in loop",results_n_sorted],
                              ["spin innermost",results_s_sorted],
                              ["gauge (that is summed over) innermost",results_g_sorted],
                              ["gauge innermost, sum explicit",results_u_sorted]]:
        print("-----")
        print(ty)
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
    
    assert all([torch.allclose(uv,uvs.transpose(4,5)),torch.allclose(uv,uvg),
                torch.allclose(uv,uvu)])

def test_throughput_matmul_gamma():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma(U,v)
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
    
    assert True


def test_throughput_matmul_gamma_shift():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_shift(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_shift(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_shift(U,v)
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
    
    assert True

def test_throughput_matmul_gamma_2shift():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift(U,v)
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
    
    assert True

def test_throughput_matmul_gamma_2shift_split():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift_split(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift_split(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift_split(U,v)
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
    
    assert True

def test_throughput_matmul_gamma_2shift_ysw():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift_ysw(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift_ysw(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2shift_ysw(U,v)
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
    
    assert True

def test_throughput_matmul_gamma_2tshift():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2tshift(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2tshift(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2tshift(U,v)
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
    
    assert True

def test_throughput_matmul_gamma_2ytshift():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn([2]+lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2ytshift(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2ytshift(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2ytshift(U,v)
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
    
    assert True


def test_throughput_matmul_simple_ytshift():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn([2]+lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_simple_ytshift(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_simple_ytshift(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_simple_ytshift(U,v)
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
    
    assert True


def test_throughput_matmul_gamma_2point():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2point(U,v)

    for _ in range(n_warmup):
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2point(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.ops.qcd_ml_accel_dirac.gauge_transform_gamma_2point(U,v)
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
    
    assert True

def test_throughput_torch_matmul():
    print()
    print("running on host", socket.gethostname())

    lat_dim = [8,8,16,16]
    print("lattice:",lat_dim)

    U = torch.randn(lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[3,4],dtype=torch.cdouble)

    n_measurements = 9000
    n_warmup = 500

    vp = torch.matmul(U,v)

    for _ in range(n_warmup):
        vp = torch.matmul(U,v)

    results = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        vp = torch.matmul(U,v)
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
    
    assert True
