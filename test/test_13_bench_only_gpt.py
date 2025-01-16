import socket
import numpy as np
import time
import copy


# This test is to generate test data for multiple different lattice sizes
# the difference to test_12 is that here, the time measurements for the different volumes
# are taken alternatingly instead of one volume at a time, to reduce errors in the computer

import gpt as g # type: ignore

print("test_throughput_wilson_alternating_volumes")
print()
print("running on host", socket.gethostname())

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


rng = g.random("sizetest")

variants = ["gpt",]

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

for nr in range(n_measurements):
    for vo,lat_dim in zip(vols,lat_dims):
    
        #print("=====")
        #print("testing lattice dimensions",lat_dim)

        U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_g[0].grid
        v_g = rng.cnormal(g.vspincolor(grid))
        dst_g = g.lattice(v_g)
        
        kappa = 1.0/2.0/(mass + 4.0)

        dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )


        for _ in range(n_warmup):
            dst_g = dw_g(v_g)


        start = time.perf_counter()
        dst_g = dw_g(v_g)
        stop = time.perf_counter()
        results["gpt"][vo][nr] = stop - start


        if nr == 0:
            start = time.perf_counter()
            stop = time.perf_counter()
            bias[nr] = stop - start


for vari in variants:
    res_all = {vo:results[vari][vo] for vo in vols}
    results_sorted_all = {vo:np.sort(results[vari][vo])[:(n_measurements // 5)] for vo in vols}

    for vo in vols:
        res = res_all[vo]
        results_sorted = results_sorted_all[vo]

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


        time_means_best[vari].append(np.mean(results_sorted) *10**6)
        time_stdevs_all[vari].append((np.std(res) + np.mean(bias)) *10**6)
        time_stdevs_best[vari].append((np.std(results_sorted) + np.mean(bias)) *10**6)
        time_peaks[vari].append(results_sorted[0] *10**6)
        
    

print("lattice_dimensions_used =", lat_dims)
print("lattice_volumes =", vols)
for n in variants:
    print(f"{n}_time_means_best_20p_in_us =", time_means_best[n])
    print(f"{n}_time_std_best_20p_in_us =", time_stdevs_best[n])
    print(f"{n}_time_std_all_in_us =", time_stdevs_all[n])
    print(f"{n}_time_peaks_in_us =", time_peaks[n])


print("all_measurements_for_gpt_in_us =", {vo:results["gpt"][vo] *10**6 for vo in vols})


print("test_throughput_wilson_clover_alternating_volumes")
print()
print("running on host", socket.gethostname())

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

rng = g.random("sizetest")

variants = ["gpt",]

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

for nr in range(n_measurements):
    for vo,lat_dim in zip(vols,lat_dims):
    
        #print("=====")
        #print("testing lattice dimensions",lat_dim)

        U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_g[0].grid
        v_g = rng.cnormal(g.vspincolor(grid))
        dst_g = g.lattice(v_g)
        
        kappa = 1.0/2.0/(mass + 4.0)

        dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )


        for _ in range(n_warmup):
            dst_g = dw_g(v_g)



        start = time.perf_counter()
        dst_g = dw_g(v_g)
        stop = time.perf_counter()
        results["gpt"][vo][nr] = stop - start


        if nr == 0:
            start = time.perf_counter()
            stop = time.perf_counter()
            bias[nr] = stop - start


for vari in variants:
    res_all = {vo:results[vari][vo] for vo in vols}
    results_sorted_all = {vo:np.sort(results[vari][vo])[:(n_measurements // 5)] for vo in vols}

    for vo in vols:
        res = res_all[vo]
        results_sorted = results_sorted_all[vo]

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


        time_means_best[vari].append(np.mean(results_sorted) *10**6)
        time_stdevs_all[vari].append((np.std(res) + np.mean(bias)) *10**6)
        time_stdevs_best[vari].append((np.std(results_sorted) + np.mean(bias)) *10**6)
        time_peaks[vari].append(results_sorted[0] *10**6)
        
    

print("lattice_dimensions_used =", lat_dims)
print("lattice_volumes =", vols)
for n in variants:
    print(f"{n}_time_means_best_20p_in_us =", time_means_best[n])
    print(f"{n}_time_std_best_20p_in_us =", time_stdevs_best[n])
    print(f"{n}_time_std_all_in_us =", time_stdevs_all[n])
    print(f"{n}_time_peaks_in_us =", time_peaks[n])


print("all_measurements_for_gpt_in_us =", {vo:results["gpt"][vo] *10**6 for vo in vols})

assert True

