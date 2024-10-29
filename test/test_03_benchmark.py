import torch
import torch.utils.benchmark as benchmark
import socket
import time
import numpy as np

import qcd_ml_accel_dirac

# test theoretical performance of the computer with some simple computations

def test_max_flops_with_matmul():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    # use matrix multiplication as a test for the theoretical performance

    m1 = torch.randn([12*16,87],dtype=torch.cdouble)
    m2 = torch.randn([87,8*8*8],dtype=torch.cdouble)

    for tn in range(1,num_threads+1):
        t0 = benchmark.Timer(
            stmt='torch.matmul(m1,m2)',
            globals={'m1': m1, 'm2': m2},
            num_threads=tn
        )
        print(t0.timeit(1000))
    
    print("=========================\n")

    assert True

def test_max_flops_with_muladd():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    # use a[i] * b[i] + c as a test for the theoretical performance

    a = torch.randn([2076 * 8**3 * 16],dtype=torch.cdouble)
    b = torch.randn([2076 * 8**3 * 16],dtype=torch.cdouble)

    for tn in range(1,num_threads+1):
        t0 = benchmark.Timer(
            # takes too long to be usable as test
            # stmt='for i in range(2076 * 8**3 * 16): a[i] * b[i] + 1.2',
            stmt='torch.mul(a,b)+1.2',
            globals={'a': a, 'b': b},
            num_threads=tn
        )
        print(t0.timeit(30))
    
    print("=========================\n")

    assert True


def test_memory_throughput_with_muladd():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    #num_threads = 1

    a = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)
    b = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)
    c = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)

    t0 = benchmark.Timer(
        stmt = 'torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar(a,b,c)',
        setup = 'import qcd_ml_accel_dirac',
        globals = {'a': a, 'b': b, 'c': c},
        num_threads = num_threads
    )
    t1 = benchmark.Timer(
        stmt = 'torch.ops.qcd_ml_accel_dirac.muladd_bench_par(a,b,c)',
        setup = 'import qcd_ml_accel_dirac',
        globals = {'a': a, 'b': b, 'c': c},
        num_threads = num_threads
    )
    t2 = benchmark.Timer(
        stmt = 'qcd_ml_accel_dirac.muladd_bench_nopar_py(a,b,c)',
        setup = 'import qcd_ml_accel_dirac',
        globals = {'a': a, 'b': b, 'c': c},
        num_threads = num_threads
    )
    t3 = benchmark.Timer(
        stmt = 'qcd_ml_accel_dirac.muladd_bench_par_py(a,b,c)',
        setup = 'import qcd_ml_accel_dirac',
        globals = {'a': a, 'b': b, 'c': c},
        num_threads = num_threads
    )

    print(t0.timeit(30000))
    print(t1.timeit(30000))
    print(t2.timeit(30000))
    print(t3.timeit(30000))

    res_py = qcd_ml_accel_dirac.muladd_bench_nopar_py(a,b,c)
    res_cpp = torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar(a,b,c)

    assert torch.allclose(res_py,res_cpp)



def test_throughput_muladd_py_timer():
    print()
    print("running on host", socket.gethostname())

    #numels = [4**4*4*3 * 2**i for i in range(9)]
    numels = [2*4*8*8*4*2] + [4*4*8*8*4*2 * i for i in range(1,23)]

    GiBs = {"parallel":[],"not parallel":[]}
    data_MiB = []

    for si in numels:
        n_measurements = 5000 * 8**3*16*4*3 //si
        n_warmup = 100 * 8**3*16*4*3 //si

        print("=====")
        print("size:",si)
        a = torch.randn([si], dtype = torch.cdouble)
        b = torch.randn([si], dtype = torch.cdouble)
        c = torch.randn([si], dtype = torch.cdouble)

        abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_par(a,b,c)
        abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar(a,b,c)

        for _ in range(n_warmup):
            abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_par(a,b,c)
            abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar(a,b,c)


        results_par = np.zeros(n_measurements)
        results_nopar = np.zeros(n_measurements)
        bias = np.zeros(n_measurements)

        for i in range(n_measurements):
            start = time.perf_counter_ns()
            abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_par(a,b,c)
            stop = time.perf_counter_ns()
            results_par[i] = stop - start

            start = time.perf_counter_ns()
            abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar(a,b,c)
            stop = time.perf_counter_ns()
            results_nopar[i] = stop - start

            start = time.perf_counter_ns()
            stop = time.perf_counter_ns()
            bias[i] = stop - start

        results_par_sorted = np.sort(results_par)[:(n_measurements // 5)]
        results_nopar_sorted = np.sort(results_nopar)[:(n_measurements // 5)]

        for par,results_sorted in [["parallel",results_par_sorted],["not parallel",results_nopar_sorted]]:
            print("-----")
            print(par)
            print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
            print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
            print(f"best : [us] {results_sorted[0]/1000}")
            print(f"mean bias : [us] {np.mean(bias)/1000}")
            print(f"std bias : [us] {np.mean(bias)/1000}")

            data_size = 4 * a.element_size() * a.nelement()
            data_size_MiB = data_size / 1024**2

            print()
            print(f"data : [MiB] {data_size_MiB: .3f}")

            throughput = data_size / (np.mean(results_sorted) / 1000**3)
            throughput_GiBs = throughput / 1024 ** 3
            throughput_peak = data_size / (results_sorted[0] / 1000**3)
            throughput_peak_GiBs = throughput_peak / 1024 ** 3

            print(f"throughput : [GiB/s] {throughput_GiBs: .3f}")
            print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")
            GiBs[par].append(throughput_GiBs)
            if par == "parallel":
                data_MiB.append(data_size_MiB)
    
    print("=======")
    print(numels)
    print(data_MiB)
    print(GiBs)
    assert True


def test_throughput_muladd_cpp_timer():
    print()
    print("running on host", socket.gethostname())
    n_measurements = 20000
    n_warmup = 400

    a = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)
    b = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)
    c = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)

    abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_par_time(a,b,c)
    abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar_time(a,b,c)

    for _ in range(n_warmup):
        abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_par_time(a,b,c)
        abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar_time(a,b,c)


    results_par = np.zeros(n_measurements)
    results_nopar = np.zeros(n_measurements)

    for i in range(n_measurements):
        abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_par_time(a,b,c)
        results_par[i] = abc[0,0,0,0,0,0]

        abc = torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar_time(a,b,c)
        results_nopar[i] = abc[0,0,0,0,0,0]


    results_par_sorted = np.sort(results_par)[:(n_measurements // 5)]
    results_nopar_sorted = np.sort(results_nopar)[:(n_measurements // 5)]

    for par,results_sorted in [["parallel",results_par_sorted],["not parallel",results_nopar_sorted]]:
        print("-----")
        print(par)
        print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
        print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
        print(f"best : [us] {results_sorted[0]/1000}")

        data_size = 4 * a.element_size() * a.nelement()
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


