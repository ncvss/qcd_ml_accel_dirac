import torch
import socket
import numpy as np
import time
import pytest

import qcd_ml_accel_dirac
import qcd_ml


try:
    import gpt as g # type: ignore
    import qcd_ml_accel_dirac.compat

    # try to call the dirac wilson to see if the c++ function was compiled
    U_try = torch.zeros([4,2,2,2,2,3,3], dtype=torch.cdouble)
    v_try = torch.zeros([2,2,2,2,4,3], dtype=torch.cdouble)
    dw_try = qcd_ml_accel_dirac.dirac_wilson_avx(U_try, -0.5)
    dwv_try = dw_try(v_try)

    @pytest.mark.benchmark
    def test_benchmark_wilson_avx():
        print()
        num_threads = torch.get_num_threads()
        print("running on host", socket.gethostname())
        print(f"Machine has {num_threads} threads")

        n_measurements = 500
        n_warmup = 20

        lat_dim = [16,16,16,32]
        print("lattice dimensions:",lat_dim)

        rng = g.random("testbm05")
        U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_g[0].grid
        v_g = rng.cnormal(g.vspincolor(grid))
        dst_g = g.lattice(v_g)

        mass = -0.5
        print("mass_parameter =",mass)
        kappa = 1.0/2.0/(mass + 4.0)

        dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

        U = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(U_g))
        v = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(v_g))
    
        dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
        dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)
        dw_avx = qcd_ml_accel_dirac.dirac_wilson_avx(U,mass)

        op_names = ["gpt", "qcd_ml", "qcd_ml_accel_dirac", "qcd_ml_accel_dirac-avx"]
        op_calls = [dw_g, dw_py, dw_cpp, dw_avx]
        op_input = [v_g, v, v, v]

        results = {name:np.zeros(n_measurements) for name in op_names}
        bias = np.zeros(n_measurements)

        for i in range(n_warmup):
            for j in range(len(op_names)):
                output = op_calls[j](op_input[j])

        for i in range(n_measurements):
            for j in range(len(op_names)):
                start = time.perf_counter_ns()
                output = op_calls[j](op_input[j])
                stop = time.perf_counter_ns()
                results[op_names[j]][i] = stop - start

            start = time.perf_counter_ns()
            stop = time.perf_counter_ns()
            bias[i] = stop - start

        results_sorted = {name:np.sort(results[name])[:(n_measurements//5)] for name in op_names}

        base_data = v.element_size() * v.nelement() * 2 + U.element_size() * U.nelement()
        hop_data = dw_avx.hop_inds.element_size() * dw_avx.hop_inds.nelement()
        datasizes = {"qcd_ml": base_data, "gpt": base_data, "qcd_ml_accel_dirac": base_data,
                     "qcd_ml_accel_dirac-avx": base_data+hop_data}

        print("===measurements===")
        print(f"mean bias : [us] {np.mean(bias)/1000}")
        print(f"std bias : [us] {np.mean(bias)/1000}")

        for name,res in results_sorted.items():
            print("-------")
            print(name)
            print(f"mean (top 20%): [us] {np.mean(res)/1000: .2f}")
            print(f"std (top 20%): [us] {np.std(res)/1000: .2f}")
            print(f"best : [us] {res[0]/1000}")

            data_size = datasizes[name]
            data_size_MiB = datasizes[name] / 1024**2

            print()
            print(f"data : [MiB] {data_size_MiB: .3f}")

            throughput = data_size / (np.mean(res) / 1000**3)
            throughput_GiBs = throughput / 1024 ** 3
            throughput_peak = data_size / (res[0] / 1000**3)
            throughput_peak_GiBs = throughput_peak / 1024 ** 3

            print(f"throughput : [GiB/s] {throughput_GiBs: .3f}")
            print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")
        

        dst_g = dw_g(v_g)
        dwv_py = dw_py(v)
        dwv_cpp = dw_cpp(v)
        dwv_avx = dw_avx(v)
        dst_torch = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(dst_g))
        assert all([torch.allclose(dwv_cpp,dwv_py),torch.allclose(dwv_cpp,dst_torch),
                    torch.allclose(dwv_avx,dwv_py)])
    

    @pytest.mark.benchmark
    def test_benchmark_wilson_clover_avx():
        print()
        num_threads = torch.get_num_threads()
        print("running on host", socket.gethostname())
        print(f"Machine has {num_threads} threads")

        n_measurements = 500
        n_warmup = 20

        lat_dim = [16,16,16,32]
        print("lattice dimensions:",lat_dim)

        rng = g.random("testbm05")
        U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_g[0].grid
        v_g = rng.cnormal(g.vspincolor(grid))
        dst_g = g.lattice(v_g)

        mass = -0.5
        print("mass_parameter =",mass)
        kappa = 1.0/2.0/(mass + 4.0)
        csw = 1.0
        print("csw =", csw)

        dwc_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

        U = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(U_g))
        v = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(v_g))
    
        dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
        dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)
        dwc_avx = qcd_ml_accel_dirac.dirac_wilson_clover_avx(U,mass,csw)

        op_names = ["gpt", "qcd_ml", "qcd_ml_accel_dirac", "qcd_ml_accel_dirac-avx"]
        op_calls = [dwc_g, dwc_py, dwc_cpp, dwc_avx]
        op_input = [v_g, v, v, v]

        results = {name:np.zeros(n_measurements) for name in op_names}
        bias = np.zeros(n_measurements)

        for i in range(n_warmup):
            for j in range(len(op_names)):
                output = op_calls[j](op_input[j])

        for i in range(n_measurements):
            for j in range(len(op_names)):
                start = time.perf_counter_ns()
                output = op_calls[j](op_input[j])
                stop = time.perf_counter_ns()
                results[op_names[j]][i] = stop - start

            start = time.perf_counter_ns()
            stop = time.perf_counter_ns()
            bias[i] = stop - start

        results_sorted = {name:np.sort(results[name])[:(n_measurements//5)] for name in op_names}

        base_data = v.element_size() * v.nelement() * 2 + U.element_size() * U.nelement()
        hop_data = dwc_avx.hop_inds.element_size() * dwc_avx.hop_inds.nelement()
        clover_data = len(dwc_cpp.field_strength) * dwc_cpp.field_strength[0].element_size() * dwc_cpp.field_strength[0].nelement()
        gridclover_data = dwc_avx.field_strength_sigma.element_size() * dwc_avx.field_strength_sigma.nelement()
        datasizes = {"qcd_ml": base_data,
                     "gpt": base_data+gridclover_data,
                     "qcd_ml_accel_dirac": base_data+clover_data,
                     "qcd_ml_accel_dirac-avx": base_data+hop_data+gridclover_data}

        print("===measurements===")
        print(f"mean bias : [us] {np.mean(bias)/1000}")
        print(f"std bias : [us] {np.mean(bias)/1000}")

        for name,res in results_sorted.items():
            print("-------")
            print(name)
            print(f"mean (top 20%): [us] {np.mean(res)/1000: .2f}")
            print(f"std (top 20%): [us] {np.std(res)/1000: .2f}")
            print(f"best : [us] {res[0]/1000}")

            data_size = datasizes[name]
            data_size_MiB = datasizes[name] / 1024**2

            print()
            print(f"data : [MiB] {data_size_MiB: .3f}")

            throughput = data_size / (np.mean(res) / 1000**3)
            throughput_GiBs = throughput / 1024 ** 3
            throughput_peak = data_size / (res[0] / 1000**3)
            throughput_peak_GiBs = throughput_peak / 1024 ** 3

            print(f"throughput : [GiB/s] {throughput_GiBs: .3f}")
            print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")
        

        dst_g = dwc_g(v_g)
        dwv_py = dwc_py(v)
        dwv_cpp = dwc_cpp(v)
        dwv_avx = dwc_avx(v)
        dst_torch = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(dst_g))
        assert all([torch.allclose(dwv_cpp,dwv_py),torch.allclose(dwv_cpp,dst_torch),
                    torch.allclose(dwv_avx,dwv_py)])


except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_avx():
        pass

    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_clover_avx():
        pass

except RuntimeError:

    @pytest.mark.skip("missing AVX")
    def test_throughput_wilson_avx():
        pass

    @pytest.mark.skip("missing AVX")
    def test_throughput_wilson_clover_avx():
        pass
