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

        n_measurements = 200
        n_warmup = 10

        lat_dim = [16,8,8,16]
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

        dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
        dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)
        dw_eo = qcd_ml_accel_dirac.dirac_wilson_eo(U,mass)

        U_l = torch.permute(U,(1,2,3,4,0,5,6,)).contiguous()
        v_l = torch.permute(v,(0,1,2,3,5,4)).contiguous()

        dw_l = qcd_ml_accel_dirac.dirac_wilson_lookup(U_l,mass)
        #dw_l = lambda v: v

        emask = dw_eo.emask
        omask = dw_eo.omask
        # keep in mind that this flattens the spatial dimensions, where the mask is applied
        ve = v[emask]
        vo = v[omask]

        # output for the dirac wilson to write to
        dwv_w = torch.empty_like(v_l)

        dwv_py = dw_py(v)
        dwv_cpp = dw_cpp(v)
        dst_g = dw_g(v_g)
        dwv_eo = dw_eo(ve,vo)
        dwv_l = dw_l(v_l)
        dw_l.write_call(v_l,dwv_w)

        hop_datasize = dw_l.hop_inds.element_size()*dw_l.hop_inds.nelement()
        print("lookup data size:", hop_datasize/(1024**2),"MiB")
        lookupdata = {"qcd_ml":0, "qcd_ml_accel_dirac":0, "gpt":0, "even-odd":0,
                      "lookup":hop_datasize, "lookup write":hop_datasize}

        #dw_l = lambda v: dwv_l

        dst_torch = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(dst_g))

        for _ in range(n_warmup):
            dwv_py = dw_py(v)
            dwv_cpp = dw_cpp(v)
            dst_g = dw_g(v_g)
            dwv_eo = dw_eo(ve,vo)
            dwv_l = dw_l(v_l)
            dw_l.write_call(v_l,dwv_w)

        #results_py = np.zeros(n_measurements)
        results_cpp = np.zeros(n_measurements)
        results_g = np.zeros(n_measurements)
        results_eo = np.zeros(n_measurements)
        results_l = np.zeros(n_measurements)
        results_w = np.zeros(n_measurements)
        bias = np.zeros(n_measurements)

        for i in range(n_measurements):
            # start = time.perf_counter_ns()
            # dwv_py = dw_py(v)
            # stop = time.perf_counter_ns()
            # results_py[i] = stop - start

            start = time.perf_counter_ns()
            dwv_cpp = dw_cpp(v)
            stop = time.perf_counter_ns()
            results_cpp[i] = stop - start

            start = time.perf_counter_ns()
            dst_g = dw_g(v_g)
            stop = time.perf_counter_ns()
            results_g[i] = stop - start

            start = time.perf_counter_ns()
            dwv_eo = dw_eo(ve,vo)
            stop = time.perf_counter_ns()
            results_eo[i] = stop - start

            start = time.perf_counter_ns()
            dwv_l = dw_l(v_l)
            stop = time.perf_counter_ns()
            results_l[i] = stop - start

            start = time.perf_counter_ns()
            dw_l.write_call(v_l,dwv_w)
            stop = time.perf_counter_ns()
            results_w[i] = stop - start

            start = time.perf_counter_ns()
            stop = time.perf_counter_ns()
            bias[i] = stop - start


        #results_py_sorted = np.sort(results_py)[:(n_measurements // 5)]
        results_cpp_sorted = np.sort(results_cpp)[:(n_measurements // 5)]
        results_g_sorted = np.sort(results_g)[:(n_measurements // 5)]
        results_eo_sorted = np.sort(results_eo)[:(n_measurements // 5)]
        results_l_sorted = np.sort(results_l)[:(n_measurements // 5)]
        results_w_sorted = np.sort(results_w)[:(n_measurements // 5)]


        for pack,results_sorted in [#["qcd_ml",results_py_sorted],
                                    ["qcd_ml_accel_dirac",results_cpp_sorted],
                                    ["gpt", results_g_sorted],["even-odd", results_eo_sorted],
                                    ["lookup", results_l_sorted],["lookup write", results_w_sorted]]:
            print("-----")
            print(pack)
            print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
            print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
            print(f"best : [us] {results_sorted[0]/1000}")
            print(f"mean bias : [us] {np.mean(bias)/1000}")
            print(f"std bias : [us] {np.mean(bias)/1000}")

            data_size = v.element_size() * v.nelement() * 2 + U.element_size() * U.nelement() + lookupdata[pack]
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

        dwv_eo_back = torch.zeros_like(dwv_py)
        dwv_eo_back[omask] = dwv_eo[1]
        dwv_eo_back[emask] = dwv_eo[0]

        dwv_l_back = torch.permute(dwv_l,(0,1,2,3,5,4))
        dwv_w_back = torch.permute(dwv_w,(0,1,2,3,5,4))
        
        assert all([torch.allclose(dwv_cpp,dwv_py),torch.allclose(dwv_cpp,dst_torch),
                    torch.allclose(dwv_py,dwv_eo_back),torch.allclose(dwv_py,dwv_l_back),
                    torch.allclose(dwv_py,dwv_w_back)])
    

    # def test_throughput_wilson_clover_gpt():
    #     print()
    #     num_threads = torch.get_num_threads()
    #     print("running on host", socket.gethostname())
    #     print(f'Machine has {num_threads} threads')

    #     n_measurements = 1000
    #     n_warmup = 10

    #     lat_dim = [8,8,8,16]
    #     print("lattice dimensions:",lat_dim)

    #     rng = g.random("test")
    #     U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
    #     grid = U_g[0].grid
    #     v_g = rng.cnormal(g.vspincolor(grid))
    #     dst_g = g.lattice(v_g)

    #     mass = -0.5
    #     csw = 1.0
    #     print("mass parameter:",mass)
    #     print("c_sw:",csw)
    #     kappa = 1.0/2.0/(mass + 4.0)

    #     dwc_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
    #                                         "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

    #     U = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(U_g))
    #     v = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(v_g))
        

    #     dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
    #     dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)
    #     dwc_pre = qcd_ml_accel_dirac.dirac_wilson_clover_precom(U,mass,csw)

    #     dwv_py = dwc_py(v)
    #     dwv_cpp = dwc_cpp(v)
    #     dst_g = dwc_g(v_g)
    #     dwv_pre = dwc_pre(v)

    #     for _ in range(n_warmup):
    #         dwv_py = dwc_py(v)
    #         dwv_cpp = dwc_cpp(v)
    #         dst_g = dwc_g(v_g)
    #         dwv_pre = dwc_pre(v)

    #     results_py = np.zeros(n_measurements)
    #     results_cpp = np.zeros(n_measurements)
    #     results_g = np.zeros(n_measurements)
    #     results_pre = np.zeros(n_measurements)
    #     bias = np.zeros(n_measurements)

    #     for i in range(n_measurements):
    #         start = time.perf_counter_ns()
    #         dwv_py = dwc_py(v)
    #         stop = time.perf_counter_ns()
    #         results_py[i] = stop - start

    #         start = time.perf_counter_ns()
    #         dwv_cpp = dwc_cpp(v)
    #         stop = time.perf_counter_ns()
    #         results_cpp[i] = stop - start

    #         start = time.perf_counter_ns()
    #         dst_g = dwc_g(v_g)
    #         stop = time.perf_counter_ns()
    #         results_g[i] = stop - start

    #         start = time.perf_counter_ns()
    #         dwv_pre = dwc_pre(v)
    #         stop = time.perf_counter_ns()
    #         results_pre[i] = stop - start

    #         start = time.perf_counter_ns()
    #         stop = time.perf_counter_ns()
    #         bias[i] = stop - start


    #     results_py_sorted = np.sort(results_py)[:(n_measurements // 5)]
    #     results_cpp_sorted = np.sort(results_cpp)[:(n_measurements // 5)]
    #     results_g_sorted = np.sort(results_g)[:(n_measurements // 5)]
    #     results_pre_sorted = np.sort(results_pre)[:(n_measurements // 5)]


    #     for pack,results_sorted in [["qcd_ml",results_py_sorted], ["qcd_ml_accel_dirac",results_cpp_sorted],
    #                                 ["gpt", results_g_sorted], ["precom",results_pre_sorted]]:
    #         print("-----")
    #         print(pack)
    #         print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
    #         print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
    #         print(f"best : [us] {results_sorted[0]/1000}")
    #         print(f"mean bias : [us] {np.mean(bias)/1000}")
    #         print(f"std bias : [us] {np.mean(bias)/1000}")

    #         data_size = v.element_size() * v.nelement() * 2 + U.element_size() * U.nelement() * 10/4
    #         data_size_MiB = data_size / 1024**2

    #         print()
    #         print(f"data : [MiB] {data_size_MiB: .3f}")

    #         throughput = data_size / (np.mean(results_sorted) / 1000**3)
    #         throughput_GiBs = throughput / 1024 ** 3
    #         throughput_peak = data_size / (results_sorted[0] / 1000**3)
    #         throughput_peak_GiBs = throughput_peak / 1024 ** 3

    #         print(f"throughput : [GiB/s] {throughput_GiBs: .3f}")
    #         print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")
            
    #     dst_torch = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(dst_g))
    #     assert all([torch.allclose(dwv_cpp,dwv_py),torch.allclose(dwv_cpp,dst_torch),
    #                 torch.allclose(dwv_cpp,dwv_pre)])

except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_gpt():
        pass
    
    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_clover_gpt():
        pass

