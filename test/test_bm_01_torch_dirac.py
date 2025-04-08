import torch
import torch.utils.benchmark as benchmark
import socket
import numpy as np
import time

import qcd_ml_accel_dirac
import qcd_ml



def test_torch_benchmark_wilson_vary_threads():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f"Machine has {num_threads} threads")

    size = [8,8,8,16]

    # not actual gauge field, but still viable for this test
    U = torch.randn([4]+size+[3,3],dtype=torch.cdouble)
    v = torch.randn(size+[4,3],dtype=torch.cdouble)
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



def test_torch_benchmark_wilson_clover_vary_threads():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f"Machine has {num_threads} threads")

    size = [8,8,8,16]
    print("grid size:", size)

    # not actual gauge field, but still viable for this test
    U = torch.randn([4]+size+[3,3],dtype=torch.cdouble)
    v = torch.randn(size+[4,3],dtype=torch.cdouble)
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



def test_torch_benchmark_wilson_vary_sizes():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    sizes = [[4,4,4,8],[8,4,4,8],[8,8,4,8],[8,8,8,8],[8,8,8,16],[16,8,8,16],[16,16,8,16],[16,16,16,16],
    [16,16,16,32]]

    mass = -0.5
    csw = 1.0
    tn = min(8,num_threads)

    assert_equal = []

    for si in sizes:

        print(f"grid size is {si[0]}x{si[1]}x{si[2]}x{si[3]}")

        # not actual gauge field, but still viable for this test
        U = torch.randn([4]+si+[3,3],dtype=torch.cdouble)
        v = torch.randn(si+[4,3],dtype=torch.cdouble)

        t0 = benchmark.Timer(
            stmt='dw_py(v)',
            setup='from qcd_ml.qcd.dirac import dirac_wilson; dw_py = dirac_wilson(U,m)',
            globals={'U': U, 'v': v, 'm': mass},
            num_threads=tn
        )

        t1 = benchmark.Timer(
            stmt='dw_cpp(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson; dw_cpp = dirac_wilson(U,m)',
            globals={'U': U, 'v': v, 'm': mass},
            num_threads=tn
        )

        print(t0.timeit(30))
        print(t1.timeit(30))

        # check if the computations give the same result
        dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
        dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)

        assert_equal.append(torch.allclose(dw_py(v),dw_cpp(v)))
    
    print("=========================\n")

    assert all(assert_equal)



def test_torch_benchmark_wilson_clover_vary_sizes():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f"Machine has {num_threads} threads")

    sizes = [[4,4,4,8],[8,4,4,8],[8,8,4,8],[8,8,8,8],[8,8,8,16],[16,8,8,16],[16,16,8,16],[16,16,16,16],
    [16,16,16,32]]

    mass = -0.5
    csw = 1.0
    tn = min(8,num_threads)

    assert_equal = []

    for si in sizes:

        print(f"grid size is {si[0]}x{si[1]}x{si[2]}x{si[3]}")

        # not actual gauge field, but still viable for this test
        U = torch.randn([4]+si+[3,3],dtype=torch.cdouble)
        v = torch.randn(si+[4,3],dtype=torch.cdouble)

        t2 = benchmark.Timer(
            stmt='dwc_py(v)',
            setup='from qcd_ml.qcd.dirac import dirac_wilson_clover; dwc_py = dirac_wilson_clover(U,m,c)',
            globals={'U': U, 'v': v, 'm': mass, 'c': csw},
            num_threads=tn
        )

        t3 = benchmark.Timer(
            stmt='dwc_cpp(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson_clover; dwc_cpp = dirac_wilson_clover(U,m,c)',
            globals={'U': U, 'v': v, 'm': mass, 'c': csw},
            num_threads=tn
        )

        print(t2.timeit(30))
        print(t3.timeit(30))

        # check if the computations give the same result
        dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
        dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)

        assert_equal.append(torch.allclose(dwc_py(v),dwc_cpp(v)))

    print("=========================\n")

    assert all(assert_equal)

