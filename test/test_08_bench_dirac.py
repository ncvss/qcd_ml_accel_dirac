import torch
import torch.utils.benchmark as benchmark
import socket
import numpy as np
import time

import qcd_ml_accel_dirac
import qcd_ml

def test_pytorch_timer_wilson():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    sizes = [[4,4,4,8],[8,8,8,16]]

    U = torch.randn([4]+sizes[1]+[3,3],dtype=torch.cdouble)
    v = torch.randn(sizes[1]+[4,3],dtype=torch.cdouble)
    mass = -0.5

    for tn in range(1,num_threads+1):

        t0 = benchmark.Timer(
            stmt='dw(v)',
            setup='from qcd_ml.qcd.dirac import dirac_wilson; dw = dirac_wilson(U,m)',
            globals={'U': U, 'v': v, 'm': mass},
            num_threads=tn
        )

        t1 = benchmark.Timer(
            stmt='dw_cpp(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson; dw_cpp = dirac_wilson(U,m)',
            globals={'U': U, 'v': v, 'm': mass},
            num_threads=tn
        )

        # note: only shown when enabling stdout in pytest via -s argument
        print(t0.timeit(20+20*tn))
        print(t1.timeit(100+100*tn))

    print("=========================\n")

    dw = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
    dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)

    assert torch.allclose(dw(v),dw_cpp(v))



def test_pytorch_timer_wilson_clover():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    sizes = [[4,4,4,8],[8,8,8,16]]

    U = torch.randn([4]+sizes[1]+[3,3],dtype=torch.cdouble)
    v = torch.randn(sizes[1]+[4,3],dtype=torch.cdouble)
    mass = -0.5
    csw = 1.0

    for tn in range(1,num_threads+1):

        t0 = benchmark.Timer(
            stmt='dwc(v)',
            setup='from qcd_ml.qcd.dirac import dirac_wilson_clover; dwc = dirac_wilson_clover(U,m,c)',
            globals={'U': U, 'v': v, 'm': mass, 'c': csw},
            num_threads=tn
        )

        t1 = benchmark.Timer(
            stmt='dwc_cpp(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson_clover; dwc_cpp = dirac_wilson_clover(U,m,c)',
            globals={'U': U, 'v': v, 'm': mass, 'c': csw},
            num_threads=tn
        )

        # note: only shown when enabling stdout in pytest via -s argument
        print(t0.timeit(20+20*tn))
        print(t1.timeit(100+100*tn))

    print("=========================\n")

    dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
    dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)

    assert torch.allclose(dwc(v),dwc_cpp(v))



def test_perf_counter_wilson():
    print()
    n_measurements = 1000
    n_warmup = 10

    lat_dim = [8,8,8,16]
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


    for lang_name,results_sorted in [["python",results_py_sorted], ["c++",results_cpp_sorted]]:
        print("-----")
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


def test_perf_counter_wilson_clover():
    print()
    n_measurements = 1000
    n_warmup = 10

    lat_dim = [8,8,8,16]
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


    for lang_name,results_sorted in [["python",results_py_sorted], ["c++",results_cpp_sorted]]:
        print("-----")
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


def test_wilson_blocked():
    print()

    lat_dims = [[8,8,8,8],[16,16,16,16],[32,32,32,32]]
    bls = 8
    tru = []

    for lat_dim in lat_dims:
        if lat_dim[0] > bls:
            print("=======")
            vol= 1
            for d in lat_dim:
                vol = vol * d
            print("lattice dimensions:",lat_dim)

            n_measurements = 2000 * (8**3*16) //vol
            n_warmup = 5

            U = torch.randn([4]+lat_dim+[3,3],dtype=torch.cdouble)
            v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)
            mass = -0.5
            
            print("mass parameter:",mass)
            print("block size:", bls)

            dw_bl = qcd_ml_accel_dirac.dirac_wilson_b(U,mass,bls)
            dw_li = qcd_ml_accel_dirac.dirac_wilson(U,mass)
            dw_bl2 = qcd_ml_accel_dirac.dirac_wilson_b2(U,mass,bls)

            dwv_li = dw_li(v)
            dwv_bl = dw_bl(v)
            dwv_bl2 = dw_bl2(v)

            for _ in range(n_warmup):
                dwv_li = dw_li(v)
                dwv_bl = dw_bl(v)
                dwv_bl2 = dw_bl2(v)

            results_li = np.zeros(n_measurements)
            results_bl = np.zeros(n_measurements)
            results_bl2 = np.zeros(n_measurements)
            bias = np.zeros(n_measurements)

            for i in range(n_measurements):
                start = time.perf_counter_ns()
                dwv_li = dw_li(v)
                stop = time.perf_counter_ns()
                results_li[i] = stop - start

                start = time.perf_counter_ns()
                dwv_bl = dw_bl(v)
                stop = time.perf_counter_ns()
                results_bl[i] = stop - start

                start = time.perf_counter_ns()
                dwv_bl2 = dw_bl2(v)
                stop = time.perf_counter_ns()
                results_bl2[i] = stop - start

                start = time.perf_counter_ns()
                stop = time.perf_counter_ns()
                bias[i] = stop - start


            results_li_sorted = np.sort(results_li)[:(n_measurements // 5)]
            results_bl_sorted = np.sort(results_bl)[:(n_measurements // 5)]
            results_bl2_sorted = np.sort(results_bl2)[:(n_measurements // 5)]


            for loop,results_sorted in [["linear",results_li_sorted],["blocked",results_bl2_sorted],["blocked and mu+/- split",results_bl_sorted]]:
                print("-----")
                print(loop)
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
            
            tru.append(torch.allclose(dwv_li,dwv_bl))
            tru.append(torch.allclose(dwv_li,dwv_bl2))
        

    assert all(tru)


def test_wilson_clover_no_precomputation():
    print()
    n_measurements = 1000
    n_warmup = 10

    lat_dim = [8,8,8,16]
    print("lattice dimensions:",lat_dim)

    U = torch.randn([4]+lat_dim+[3,3],dtype=torch.cdouble)
    v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)
    mass = -0.5
    csw = 1.0
    print("mass parameter:",mass)
    print("c_sw:",csw)
    
    dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover_nopre(U,mass,csw)

    dwcv_cpp = dwc_cpp(v)

    for _ in range(n_warmup):
        dwcv_cpp = dwc_cpp(v)

    results_cpp = np.zeros(n_measurements)
    bias = np.zeros(n_measurements)

    for i in range(n_measurements):
        start = time.perf_counter_ns()
        dwcv_cpp = dwc_cpp(v)
        stop = time.perf_counter_ns()
        results_cpp[i] = stop - start

        start = time.perf_counter_ns()
        stop = time.perf_counter_ns()
        bias[i] = stop - start

    results_cpp_sorted = np.sort(results_cpp)[:(n_measurements // 5)]

    for lang_name,results_sorted in [["c++",results_cpp_sorted]]:
        print("-----")
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
        

    assert True

