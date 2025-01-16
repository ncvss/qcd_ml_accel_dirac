import torch
import socket
import numpy as np
import time
import pytest
import copy

import qcd_ml_accel_dirac
import qcd_ml

# This test is to generate test data for multiple different lattice sizes
# the difference to test_12 is that here, the time measurements for the different volumes
# are taken alternatingly instead of one volume at a time, to reduce errors in the computer

try:

    def test_throughput_wilson_alternating_volumes():
        print()
        num_threads = torch.get_num_threads()
        print("running on host", socket.gethostname())
        print(f'Machine has {num_threads} threads')

        lat_dims = [[4,4,4,4]]
        for li in range(7):
            newdim = copy.copy(lat_dims[li])
            newdim[(li+3)%4] *= 2
            lat_dims.append(newdim)
        
        vols = [l[0]*l[1]*l[2]*l[3] for l in lat_dims]

        mass = -0.5
        print("mass parameter =",mass)

        n_measurements = 100
        n_warmup = 2
        print("number of time measurements =", n_measurements)

        check_correct = []

        variants = ["avx","template"]

        time_means_best = {n:[] for n in variants}
        time_stdevs_best = {n:[] for n in variants}
        time_stdevs_all = {n:[] for n in variants}
        thrpt_means_best = {n:[] for n in variants}
        thrpt_stdevs_best = {n:[] for n in variants}
        thrpt_stdevs_all = {n:[] for n in variants}
        time_peaks = {n:[] for n in variants}
        thrpt_peaks = {n:[] for n in variants}

        results = {n:{vo:np.zeros(n_measurements) for vo in vols} for n in variants}
        bias = np.zeros(n_measurements)

        data_sizes = {vari:{vo:0 for vo in vols} for vari in variants}
        
        for nr in range(n_measurements):
            for vo,lat_dim in zip(vols,lat_dims):
            
                U = torch.randn([4]+lat_dim+[3,3], dtype = torch.cdouble)
                v = torch.randn(lat_dim+[4,3], dtype = torch.cdouble)
                
            
                dw_avx = qcd_ml_accel_dirac.dirac_wilson_avx_old(U,mass)

                for _ in range(n_warmup):
                    dwv_avx = dw_avx(v)
                    dwv_t = dw_avx.template_call(v)

                start = time.perf_counter_ns()
                dwv_avx = dw_avx(v)
                stop = time.perf_counter_ns()
                results["avx"][vo][nr] = stop - start

                start = time.perf_counter_ns()
                dwv_avx = dw_avx.template_call(v)
                stop = time.perf_counter_ns()
                results["template"][vo][nr] = stop - start

                if nr == 0:
                    start = time.perf_counter_ns()
                    stop = time.perf_counter_ns()
                    bias[nr] = stop - start
                    
                    dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
                    dwv_py = dw_py(v)

                    check_correct.append(torch.allclose(dwv_py,dwv_avx))
                    check_correct.append(torch.allclose(dwv_py,dwv_t))

                    v_size = v.element_size() * v.nelement()
                    U_size = U.element_size() * U.nelement()
                    hop_size = dw_avx.hop_inds.element_size() * dw_avx.hop_inds.nelement()

                    data_sizes["avx"][vo] =  v_size * 2 + U_size + hop_size
                    data_sizes["template"][vo] =  v_size * 2 + U_size + hop_size

        for vari in variants:
            res_all = {vo:results[vari][vo] for vo in vols}
            results_sorted_all = {vo:np.sort(results[vari][vo])[:(n_measurements // 5)] for vo in vols}

            for vo in vols:
                res = res_all[vo]
                results_sorted = results_sorted_all[vo]

                data_size = data_sizes[vari][vo]
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


                time_means_best[vari].append(np.mean(results_sorted) / 1000)
                time_stdevs_all[vari].append((np.std(res) + np.mean(bias)) / 1000)
                time_stdevs_best[vari].append((np.std(results_sorted) + np.mean(bias)) / 1000)
                time_peaks[vari].append(results_sorted[0] / 1000)
                thrpt_means_best[vari].append(np.mean(throughput_GiBs_best))
                thrpt_stdevs_all[vari].append(np.std(throughput_GiBs_all))
                thrpt_stdevs_best[vari].append(np.std(throughput_GiBs_best))
                thrpt_peaks[vari].append(throughput_peak_GiBs)
                
            

        print("lattice_dimensions_used =", lat_dims)
        print("lattice_volumes =", vols)
        for n in variants:
            print(f"{n}_time_means_best_20p_in_us =", time_means_best[n])
            print(f"{n}_time_std_best_20p_in_us =", time_stdevs_best[n])
            print(f"{n}_time_std_all_in_us =", time_stdevs_all[n])
            print(f"{n}_time_peaks_in_us =", time_peaks[n])
            print(f"{n}_throughput_means_best_20p_in_GiB_per_s =", thrpt_means_best[n])
            print(f"{n}_throughput_std_best_20p_in_GiB_per_s =", thrpt_stdevs_best[n])
            print(f"{n}_throughput_std_all_in_GiB_per_s =", thrpt_stdevs_all[n])
            print(f"{n}_throughput_peaks_in_GiB_per_s =", thrpt_peaks[n])


        print("all_measurements_for_template_in_us =", {vo:results["template"][vo] / 1000 for vo in vols})

        assert all(check_correct)
    

    def test_throughput_wilson_clover_alternating_volumes():
        print()
        num_threads = torch.get_num_threads()
        print("running on host", socket.gethostname())
        print(f'Machine has {num_threads} threads')

        lat_dims = [[4,4,4,4]]
        for li in range(7):
            newdim = copy.copy(lat_dims[li])
            newdim[(li+3)%4] *= 2
            lat_dims.append(newdim)
        
        vols = [l[0]*l[1]*l[2]*l[3] for l in lat_dims]

        mass = -0.5
        print("mass parameter =",mass)
        csw = 1.0
        print("csw =",csw)

        n_measurements = 100
        n_warmup = 2
        print("number of time measurements =", n_measurements)

        check_correct = []

        variants = ["avx","gpt","template"]

        time_means_best = {n:[] for n in variants}
        time_stdevs_best = {n:[] for n in variants}
        time_stdevs_all = {n:[] for n in variants}
        thrpt_means_best = {n:[] for n in variants}
        thrpt_stdevs_best = {n:[] for n in variants}
        thrpt_stdevs_all = {n:[] for n in variants}
        time_peaks = {n:[] for n in variants}
        thrpt_peaks = {n:[] for n in variants}

        results = {n:{vo:np.zeros(n_measurements) for vo in vols} for n in variants}
        bias = np.zeros(n_measurements)

        data_sizes = {vari:{vo:0 for vo in vols} for vari in variants}
        
        for nr in range(n_measurements):
            for vo,lat_dim in zip(vols,lat_dims):
            
                U = torch.randn([4]+lat_dim+[3,3], dtype = torch.cdouble)
                v = torch.randn(lat_dim+[4,3], dtype = torch.cdouble)
                
            
                dw_avx = qcd_ml_accel_dirac.dirac_wilson_clover_avx_old(U,mass,csw)

                for _ in range(n_warmup):
                    dwv_avx = dw_avx(v)
                    dwv_t = dw_avx.template_call(v)

                start = time.perf_counter_ns()
                dwv_avx = dw_avx(v)
                stop = time.perf_counter_ns()
                results["avx"][vo][nr] = stop - start

                start = time.perf_counter_ns()
                dwv_avx = dw_avx.template_call(v)
                stop = time.perf_counter_ns()
                results["template"][vo][nr] = stop - start

                if nr == 0:
                    start = time.perf_counter_ns()
                    stop = time.perf_counter_ns()
                    bias[nr] = stop - start
                    
                    dw_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
                    dwv_py = dw_py(v)

                    check_correct.append(torch.allclose(dwv_py,dwv_avx))
                    check_correct.append(torch.allclose(dwv_py,dwv_t))

                    v_size = v.element_size() * v.nelement()
                    U_size = U.element_size() * U.nelement()
                    f_size = dw_avx.field_strength.element_size() * dw_avx.field_strength.nelement()
                    hop_size = dw_avx.hop_inds.element_size() * dw_avx.hop_inds.nelement()

                    data_sizes["avx"][vo] =  v_size * 2 + U_size + hop_size + f_size
                    data_sizes["template"][vo] =  v_size * 2 + U_size + hop_size + f_size

        for vari in variants:
            res_all = {vo:results[vari][vo] for vo in vols}
            results_sorted_all = {vo:np.sort(results[vari][vo])[:(n_measurements // 5)] for vo in vols}

            for vo in vols:
                res = res_all[vo]
                results_sorted = results_sorted_all[vo]

                data_size = data_sizes[vari][vo]
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


                time_means_best[vari].append(np.mean(results_sorted) / 1000)
                time_stdevs_all[vari].append((np.std(res) + np.mean(bias)) / 1000)
                time_stdevs_best[vari].append((np.std(results_sorted) + np.mean(bias)) / 1000)
                time_peaks[vari].append(results_sorted[0] / 1000)
                thrpt_means_best[vari].append(np.mean(throughput_GiBs_best))
                thrpt_stdevs_all[vari].append(np.std(throughput_GiBs_all))
                thrpt_stdevs_best[vari].append(np.std(throughput_GiBs_best))
                thrpt_peaks[vari].append(throughput_peak_GiBs)
                
            

        print("lattice_dimensions_used =", lat_dims)
        print("lattice_volumes =", vols)
        for n in variants:
            print(f"{n}_time_means_best_20p_in_us =", time_means_best[n])
            print(f"{n}_time_std_best_20p_in_us =", time_stdevs_best[n])
            print(f"{n}_time_std_all_in_us =", time_stdevs_all[n])
            print(f"{n}_time_peaks_in_us =", time_peaks[n])
            print(f"{n}_throughput_means_best_20p_in_GiB_per_s =", thrpt_means_best[n])
            print(f"{n}_throughput_std_best_20p_in_GiB_per_s =", thrpt_stdevs_best[n])
            print(f"{n}_throughput_std_all_in_GiB_per_s =", thrpt_stdevs_all[n])
            print(f"{n}_throughput_peaks_in_GiB_per_s =", thrpt_peaks[n])


        print("all_measurements_for_template_in_us =", {vo:results["template"][vo] / 1000 for vo in vols})

        assert all(check_correct)
    
    

except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_alternating_volumes():
        pass

    @pytest.mark.skip("missing gpt")
    def test_throughput_wilson_clover_alternating_volumes():
        pass  


