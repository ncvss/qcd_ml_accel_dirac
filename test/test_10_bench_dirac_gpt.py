import torch
import socket
import numpy as np
import time
import pytest

import qcd_ml_accel_dirac
import qcd_ml


try:
    import gpt as g # type: ignore

    def test_throughput_wilson_gpt():
        print()
        num_threads = torch.get_num_threads()
        print("running on host", socket.gethostname())
        print(f'Machine has {num_threads} threads')

        n_measurements = 1000
        n_warmup = 10

        lat_dim = [8,8,8,16]
        print("lattice dimensions:",lat_dim)

        rng = g.random("test")
        U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_g[0].grid
        v_g = rng.cnormal(g.vspincolor(grid))
        dst_g = g.lattice(v_g)

        mass = -0.5
        print("mass parameter:",mass)
        kappa = 1.0/2.0/(mass + 4.0)

        dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

        U = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(U_g))
        v = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(v_g))
        # U = torch.randn([4]+lat_dim+[3,3],dtype=torch.cdouble)
        # v = torch.randn(lat_dim+[4,3],dtype=torch.cdouble)
        

        dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
        dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)

        dwv_py = dw_py(v)
        dwv_cpp = dw_cpp(v)
        dst_g = dw_g(v_g)

        for _ in range(n_warmup):
            dwv_py = dw_py(v)
            dwv_cpp = dw_cpp(v)
            dst_g = dw_g(v_g)

        results_py = np.zeros(n_measurements)
        results_cpp = np.zeros(n_measurements)
        results_g = np.zeros(n_measurements)
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
            dst_g = dw_g(v_g)
            stop = time.perf_counter_ns()
            results_g[i] = stop - start

            start = time.perf_counter_ns()
            stop = time.perf_counter_ns()
            bias[i] = stop - start


        results_py_sorted = np.sort(results_py)[:(n_measurements // 5)]
        results_cpp_sorted = np.sort(results_cpp)[:(n_measurements // 5)]
        results_g_sorted = np.sort(results_g)[:(n_measurements // 5)]


        for pack,results_sorted in [["qcd_ml",results_py_sorted], ["qcd_ml_accel_dirac",results_cpp_sorted],
                                    ["gpt", results_g_sorted]]:
            print("-----")
            print(pack)
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
            
        dst_torch = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(dst_g))
        assert all([torch.allclose(dwv_cpp,dwv_py),torch.allclose(dwv_cpp,dst_torch)])
    

    def test_throughput_wilson_clover_gpt():
        print()
        num_threads = torch.get_num_threads()
        print("running on host", socket.gethostname())
        print(f'Machine has {num_threads} threads')

        n_measurements = 1000
        n_warmup = 10

        lat_dim = [8,8,8,16]
        print("lattice dimensions:",lat_dim)

        rng = g.random("test")
        U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_g[0].grid
        v_g = rng.cnormal(g.vspincolor(grid))
        dst_g = g.lattice(v_g)

        mass = -0.5
        csw = 1.0
        print("mass parameter:",mass)
        print("c_sw:",csw)
        kappa = 1.0/2.0/(mass + 4.0)

        dwc_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

        U = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(U_g))
        v = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(v_g))
        

        dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
        dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)

        dwv_py = dwc_py(v)
        dwv_cpp = dwc_cpp(v)
        dst_g = dwc_g(v_g)

        for _ in range(n_warmup):
            dwv_py = dwc_py(v)
            dwv_cpp = dwc_cpp(v)
            dst_g = dwc_g(v_g)

        results_py = np.zeros(n_measurements)
        results_cpp = np.zeros(n_measurements)
        results_g = np.zeros(n_measurements)
        bias = np.zeros(n_measurements)

        for i in range(n_measurements):
            start = time.perf_counter_ns()
            dwv_py = dwc_py(v)
            stop = time.perf_counter_ns()
            results_py[i] = stop - start

            start = time.perf_counter_ns()
            dwv_cpp = dwc_cpp(v)
            stop = time.perf_counter_ns()
            results_cpp[i] = stop - start

            start = time.perf_counter_ns()
            dst_g = dwc_g(v_g)
            stop = time.perf_counter_ns()
            results_g[i] = stop - start

            start = time.perf_counter_ns()
            stop = time.perf_counter_ns()
            bias[i] = stop - start


        results_py_sorted = np.sort(results_py)[:(n_measurements // 5)]
        results_cpp_sorted = np.sort(results_cpp)[:(n_measurements // 5)]
        results_g_sorted = np.sort(results_g)[:(n_measurements // 5)]


        for pack,results_sorted in [["qcd_ml",results_py_sorted], ["qcd_ml_accel_dirac",results_cpp_sorted],
                                    ["gpt", results_g_sorted]]:
            print("-----")
            print(pack)
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
            
        dst_torch = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(dst_g))
        assert all([torch.allclose(dwv_cpp,dwv_py),torch.allclose(dwv_cpp,dst_torch)])

except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_gpt():
        pass
    
    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_clover_gpt():
        pass

