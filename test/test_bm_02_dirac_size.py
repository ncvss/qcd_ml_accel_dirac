import torch
import torch.utils.benchmark as benchmark
import socket

import qcd_ml_accel_dirac
import qcd_ml

def test_different_sizes_for_wilson():
    num_threads = torch.get_num_threads()
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    sizes = [[4,4,4,8],[8,4,4,8],[8,8,4,8],[8,8,8,8],[8,8,8,16],[16,8,8,16],[16,16,8,16],[16,16,16,16],
    [16,16,16,32]]

    mass = -0.5
    csw = 1.0
    tn = min(8,num_threads)

    assert_equal = []

    for si in sizes:

        print(f"size is {si[0]}x{si[1]}x{si[2]}x{si[3]}")

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

        print(t0.timeit(30))
        print(t1.timeit(30))
        print(t2.timeit(30))
        print(t3.timeit(30))

        # check if the computations give the same result
        dw_py = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
        dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)
        dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
        dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)

        assert_equal.append(torch.allclose(dw_py(v),dw_cpp(v)))
        assert_equal.append(torch.allclose(dwc_py(v),dwc_cpp(v)))


    assert all(assert_equal)