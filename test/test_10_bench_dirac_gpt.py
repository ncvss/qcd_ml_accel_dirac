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

        emask = qcd_ml_accel_dirac.evenmask(lat_dim)
        omask = torch.logical_not(emask)
        eo_dim = lat_dim[:]
        eo_dim[-1] //= 2
        eo_size = eo_dim[0]*eo_dim[1]*eo_dim[2]*eo_dim[3]
        print("eodim", eo_dim)

        ve = v[emask]
        vo = v[omask]
        Ue = U[:,emask]
        Uo = U[:,omask]

        # Ut = torch.sum(U,(5,6))
        # Uto = Ut[:,omask].reshape([4,8,8,8,8])
        # print(Ut[0,0,2])
        # print(Uto[0,0,2])
        # sieht so aus wie erwartet

        # ve = v[mask].reshape(eo_dim+[4,3])
        # vo = torch.roll(v,shifts=-1,dims=3)[mask].reshape(eo_dim+[4,3])
        # Ue = U[:,mask].reshape([4]+eo_dim+[3,3])
        # Uo = torch.roll(U,shifts=-1,dims=4)[:,mask].reshape([4]+eo_dim+[3,3])

        print("shapes of Ue and ve:")
        print(Ue.shape)
        print(ve.shape)

        # v_back = torch.zeros_like(v)
        # v_back[mask] = vo.reshape([eo_size,4,3])
        # v_back = torch.roll(v_back,1,3)
        # v_back[mask] = ve.reshape([eo_size,4,3])

        # print("the back conversion worked:", torch.allclose(v,v_back))

        dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
        dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)
        dw_eo = qcd_ml_accel_dirac.dirac_wilson_eo(Ue,Uo,mass,eo_dim)

        dwv_py = dw_py(v)
        dwv_cpp = dw_cpp(v)
        dst_g = dw_g(v_g)
        dwv_eo = dw_eo(ve,vo)

        dst_torch = torch.tensor(qcd_ml_accel_dirac.lattice_to_array(dst_g))

        dwv_eo_back = torch.zeros_like(dwv_py)
        dwv_eo_back[omask] = dwv_eo[1]
        #dwv_eo_back = torch.roll(dwv_eo_back,1,3)
        dwv_eo_back[emask] = dwv_eo[0]
        print(dwv_eo_back.shape)
        print(dwv_eo_back[0,3,0,0])
        print(dwv_py[0,3,0,0])
        incorr = torch.tensor(torch.sum(torch.abs(dwv_eo_back-dwv_py),(4,5))>0.01,dtype=torch.int)
        # print(incorr[0,0])
        # print(incorr[1,1])
        # print(incorr[2,3])
        print(torch.sum(incorr))
        # exakt 1/4 der Matrizen ist falsch
        # genau dann falsch, wenn x+y+z odd und t odd

        assert all([torch.allclose(dwv_cpp,dwv_py),torch.allclose(dwv_cpp,dst_torch),
                    torch.allclose(dwv_py,dwv_eo_back)])


        for _ in range(n_warmup):
            dwv_py = dw_py(v)
            dwv_cpp = dw_cpp(v)
            dst_g = dw_g(v_g)
            dwv_eo = dw_eo(ve,vo)

        #results_py = np.zeros(n_measurements)
        results_cpp = np.zeros(n_measurements)
        results_g = np.zeros(n_measurements)
        results_eo = np.zeros(n_measurements)
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
            stop = time.perf_counter_ns()
            bias[i] = stop - start


        #results_py_sorted = np.sort(results_py)[:(n_measurements // 5)]
        results_cpp_sorted = np.sort(results_cpp)[:(n_measurements // 5)]
        results_g_sorted = np.sort(results_g)[:(n_measurements // 5)]
        results_eo_sorted = np.sort(results_eo)[:(n_measurements // 5)]


        for pack,results_sorted in [#["qcd_ml",results_py_sorted],
                                    ["qcd_ml_accel_dirac",results_cpp_sorted],
                                    ["gpt", results_g_sorted],["even-odd", results_eo_sorted]]:
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

        dwv_eo_back = torch.zeros_like(dwv_py)
        dwv_eo_back[omask] = dwv_eo[1]
        #dwv_eo_back = torch.roll(dwv_eo_back,1,3)
        dwv_eo_back[emask] = dwv_eo[0]
        

        assert all([torch.allclose(dwv_cpp,dwv_py),torch.allclose(dwv_cpp,dst_torch),
                    torch.allclose(dwv_py,dwv_eo_back)])
    

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

