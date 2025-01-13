import torch
import socket
import numpy as np
import time
import pytest
import copy

import qcd_ml_accel_dirac
import qcd_ml

# This test is to generate test data for multiple different lattice sizes

try:
    import gpt as g # type: ignore
    import qcd_ml_accel_dirac.compat

    def test_throughput_wilson_different_volumes():
        print()
        num_threads = torch.get_num_threads()
        print("running on host", socket.gethostname())
        print(f'Machine has {num_threads} threads')

        lat_dims = [[4,4,4,4]]
        for li in range(10):
            newdim = copy.copy(lat_dims[li])
            newdim[(li+3)%4] *= 2
            lat_dims.append(newdim)
        
        vols = [l[0]*l[1]*l[2]*l[3] for l in lat_dims]

        mass = -0.5
        print("mass parameter =",mass)

        n_measurements = 5000
        n_warmup = 50
        print("number of time measurements =", n_measurements)

        check_correct = []

        rng = g.random("sizetest")

        time_means_best = {"avx":[], "gpt":[]}
        time_stdevs_best = {"avx":[], "gpt":[]}
        time_stdevs_all = {"avx":[], "gpt":[]}
        thrpt_means_best = {"avx":[], "gpt":[]}
        thrpt_stdevs_best = {"avx":[], "gpt":[]}
        thrpt_stdevs_all = {"avx":[], "gpt":[]}

        for lat_dim in lat_dims:

            #print("=====")
            print("testing lattice dimensions",lat_dim)

            U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
            grid = U_g[0].grid
            v_g = rng.cnormal(g.vspincolor(grid))
            dst_g = g.lattice(v_g)
            
            kappa = 1.0/2.0/(mass + 4.0)

            dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
                                                "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

            U = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(U_g))
            v = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(v_g))

            Unew = torch.permute(U,(1,2,3,4,0,5,6)).contiguous()
            vnew = torch.permute(v,(0,1,2,3,5,4)).contiguous()
            
            dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
            dwv_py = dw_py(v)

            dw_avx = qcd_ml_accel_dirac.dirac_wilson_avx(Unew,mass)

            dst_g = dw_g(v_g)
            dwv_avx = dw_avx(vnew)

            for _ in range(n_warmup):
                dst_g = dw_g(v_g)
                dwv_avx = dw_avx(vnew)

            results_avx = np.zeros(n_measurements)
            results_g = np.zeros(n_measurements)
            bias = np.zeros(n_measurements)

            for i in range(n_measurements):
                start = time.perf_counter_ns()
                dwv_avx = dw_avx(vnew)
                stop = time.perf_counter_ns()
                results_avx[i] = stop - start

                start = time.perf_counter_ns()
                dst_g = dw_g(v_g)
                stop = time.perf_counter_ns()
                results_g[i] = stop - start

                start = time.perf_counter_ns()
                stop = time.perf_counter_ns()
                bias[i] = stop - start


            results_avx_sorted = np.sort(results_avx)[:(n_measurements // 5)]
            results_g_sorted = np.sort(results_g)[:(n_measurements // 5)]
            
            # size of the additional hop term lookup table (only for dw_avx)
            hop_size = dw_avx.hop_inds.element_size() * dw_avx.hop_inds.nelement()
            hopdata = {"avx":hop_size, "gpt":0}

            for pack,results_sorted,res in [["gpt", results_g_sorted, results_g],
                                          ["avx", results_avx_sorted, results_avx]]:
                
                data_size = v.element_size() * v.nelement() * 2 + U.element_size() * U.nelement() + hopdata[pack]
                data_size_MiB = data_size / 1024**2

                throughput_best = data_size / (results_sorted / 1000**3)
                throughput_GiBs_best = throughput_best / 1024 ** 3

                throughput_all = data_size / (res / 1000**3)
                throughput_GiBs_all = throughput_all / 1024 ** 3

                throughput_peak = data_size / (results_sorted[0] / 1000**3)
                throughput_peak_GiBs = throughput_peak / 1024 ** 3

                # print("-----")
                # print(pack)
                # print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
                # print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
                # print(f"best : [us] {results_sorted[0]/1000}")
                # print(f"mean bias : [us] {np.mean(bias)/1000}")
                # print(f"std bias : [us] {np.mean(bias)/1000}")

                # print()
                # print(f"data : [MiB] {data_size_MiB: .3f}")

                # print(f"throughput : [GiB/s] {np.mean(throughput_GiBs): .3f}")
                # print(f"std : [GiB/s] {np.std(throughput_GiBs)}")
                # print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")


                time_means_best[pack].append(np.mean(results_sorted) / 1000)
                time_stdevs_all[pack].append((np.std(res) + np.mean(bias)) / 1000)
                time_stdevs_best[pack].append((np.std(results_sorted) + np.mean(bias)) / 1000)
                thrpt_means_best[pack].append(np.mean(throughput_GiBs_best))
                thrpt_stdevs_all[pack].append(np.std(throughput_GiBs_all))
                thrpt_stdevs_best[pack].append(np.std(throughput_GiBs_best))
    
                
            dst_torch = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(dst_g))
            dwv_avx_old = torch.permute(dwv_avx, (0,1,2,3,5,4))
            check_correct.append(torch.allclose(dwv_py,dwv_avx_old))
            check_correct.append(torch.allclose(dwv_py,dst_torch))

        print("lattice_dimensions_used =", lat_dims)
        print("lattice_volumes =", vols)
        print("avx_time_means_best_20p_in_us =", time_means_best["avx"])
        print("gpt_time_means_best_20p_in_us =", time_means_best["gpt"])
        print("avx_time_std_best_20p_in_us =", time_stdevs_best["avx"])
        print("gpt_time_std_best_20p_in_us =", time_stdevs_best["gpt"])
        print("avx_time_std_all_in_us =", time_stdevs_all["avx"])
        print("gpt_time_std_all_in_us =", time_stdevs_all["gpt"])
        print("avx_throughput_means_best_20p_in_GiB_per_s =", thrpt_means_best["avx"])
        print("gpt_throughput_means_best_20p_in_GiB_per_s =", thrpt_means_best["gpt"])
        print("avx_throughput_errors_best_20p_in_GiB_per_s =", thrpt_stdevs_best["avx"])
        print("gpt_throughput_errors_best_20p_in_GiB_per_s =", thrpt_stdevs_best["gpt"])
        print("avx_throughput_errors_all_in_GiB_per_s =", thrpt_stdevs_all["avx"])
        print("gpt_throughput_errors_all_in_GiB_per_s =", thrpt_stdevs_all["gpt"])

        assert all(check_correct)
    

    def test_throughput_wilson_clover_different_volumes():
        print()
        num_threads = torch.get_num_threads()
        print("running on host", socket.gethostname())
        print(f'Machine has {num_threads} threads')

        lat_dims = [[4,4,4,4]]
        for li in range(5):
            newdim = copy.copy(lat_dims[li])
            newdim[(li+3)%4] *= 2
            lat_dims.append(newdim)
        
        vols = [l[0]*l[1]*l[2]*l[3] for l in lat_dims]

        mass = -0.5
        print("mass parameter =",mass)
        csw = 1.0
        print("csw =", csw)

        n_measurements = 5000
        n_warmup = 50
        print("number of time measurements =", n_measurements)

        check_correct = []

        rng = g.random("clovertest")

        time_means_best = {"avx":[], "gpt":[]}
        time_stdevs_best = {"avx":[], "gpt":[]}
        time_stdevs_all = {"avx":[], "gpt":[]}
        thrpt_means_best = {"avx":[], "gpt":[]}
        thrpt_stdevs_best = {"avx":[], "gpt":[]}
        thrpt_stdevs_all = {"avx":[], "gpt":[]}

        for lat_dim in lat_dims:

            #print("=====")
            print("testing lattice dimensions",lat_dim)

            U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
            grid = U_g[0].grid
            v_g = rng.cnormal(g.vspincolor(grid))
            dst_g = g.lattice(v_g)
            
            kappa = 1.0/2.0/(mass + 4.0)

            dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
                                                "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

            U = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(U_g))
            v = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(v_g))

            
            dw_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
            dwv_py = dw_py(v)

            dw_avx = qcd_ml_accel_dirac.dirac_wilson_clover_avx_old(U,mass,csw)

            dst_g = dw_g(v_g)
            dwv_avx = dw_avx.template_call(v)

            for _ in range(n_warmup):
                dst_g = dw_g(v_g)
                dwv_avx = dw_avx.template_call(v)

            results_avx = np.zeros(n_measurements)
            results_g = np.zeros(n_measurements)
            bias = np.zeros(n_measurements)

            for i in range(n_measurements):
                start = time.perf_counter_ns()
                dwv_avx = dw_avx.template_call(v)
                stop = time.perf_counter_ns()
                results_avx[i] = stop - start

                start = time.perf_counter_ns()
                dst_g = dw_g(v_g)
                stop = time.perf_counter_ns()
                results_g[i] = stop - start

                start = time.perf_counter_ns()
                stop = time.perf_counter_ns()
                bias[i] = stop - start


            results_avx_sorted = np.sort(results_avx)[:(n_measurements // 5)]
            results_g_sorted = np.sort(results_g)[:(n_measurements // 5)]
            
            # size of the additional hop term lookup table (only for dw_avx)
            hop_size = dw_avx.hop_inds.element_size() * dw_avx.hop_inds.nelement()
            hopdata = {"avx":hop_size, "gpt":0}

            for pack,results_sorted,res in [["gpt", results_g_sorted, results_g],
                                          ["avx", results_avx_sorted, results_avx]]:
                
                data_size = v.element_size() * v.nelement() * 2 + U.element_size() * U.nelement() + hopdata[pack]
                data_size_MiB = data_size / 1024**2

                throughput_best = data_size / (results_sorted / 1000**3)
                throughput_GiBs_best = throughput_best / 1024 ** 3

                throughput_all = data_size / (res / 1000**3)
                throughput_GiBs_all = throughput_all / 1024 ** 3

                throughput_peak = data_size / (results_sorted[0] / 1000**3)
                throughput_peak_GiBs = throughput_peak / 1024 ** 3

                # print("-----")
                # print(pack)
                # print(f"mean (top 20%): [us] {np.mean(results_sorted)/1000: .2f}")
                # print(f"std (top 20%): [us] {np.std(results_sorted)/1000: .2f}")
                # print(f"best : [us] {results_sorted[0]/1000}")
                # print(f"mean bias : [us] {np.mean(bias)/1000}")
                # print(f"std bias : [us] {np.mean(bias)/1000}")

                # print()
                # print(f"data : [MiB] {data_size_MiB: .3f}")

                # print(f"throughput : [GiB/s] {np.mean(throughput_GiBs): .3f}")
                # print(f"std : [GiB/s] {np.std(throughput_GiBs)}")
                # print(f"peak thrpt. : [GiB/s] {throughput_peak_GiBs: .3f}")


                time_means_best[pack].append(np.mean(results_sorted) / 1000)
                time_stdevs_all[pack].append((np.std(res) + np.mean(bias)) / 1000)
                time_stdevs_best[pack].append((np.std(results_sorted) + np.mean(bias)) / 1000)
                thrpt_means_best[pack].append(np.mean(throughput_GiBs_best))
                thrpt_stdevs_all[pack].append(np.std(throughput_GiBs_all))
                thrpt_stdevs_best[pack].append(np.std(throughput_GiBs_best))
    
                
            dst_torch = torch.tensor(qcd_ml_accel_dirac.compat.lattice_to_array(dst_g))
            check_correct.append(torch.allclose(dwv_py,dwv_avx))
            check_correct.append(torch.allclose(dwv_py,dst_torch))

        print("lattice_dimensions_used =", lat_dims)
        print("lattice_volumes =", vols)
        print("avx_time_means_best_20p_in_us =", time_means_best["avx"])
        print("gpt_time_means_best_20p_in_us =", time_means_best["gpt"])
        print("avx_time_std_best_20p_in_us =", time_stdevs_best["avx"])
        print("gpt_time_std_best_20p_in_us =", time_stdevs_best["gpt"])
        print("avx_time_std_all_in_us =", time_stdevs_all["avx"])
        print("gpt_time_std_all_in_us =", time_stdevs_all["gpt"])
        print("avx_throughput_means_best_20p_in_GiB_per_s =", thrpt_means_best["avx"])
        print("gpt_throughput_means_best_20p_in_GiB_per_s =", thrpt_means_best["gpt"])
        print("avx_throughput_errors_best_20p_in_GiB_per_s =", thrpt_stdevs_best["avx"])
        print("gpt_throughput_errors_best_20p_in_GiB_per_s =", thrpt_stdevs_best["gpt"])
        print("avx_throughput_errors_all_in_GiB_per_s =", thrpt_stdevs_all["avx"])
        print("gpt_throughput_errors_all_in_GiB_per_s =", thrpt_stdevs_all["gpt"])

        assert all(check_correct)
    

except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_different_volumes():
        pass

    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_clover_different_volumes():
        pass  


