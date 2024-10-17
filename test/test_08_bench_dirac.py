import torch
import torch.utils.benchmark as benchmark
import socket

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
        print(t1.timeit(20+20*tn))

    print("=========================\n")

    dw = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
    dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)

    assert torch.allclose(dw(v),dw_cpp(v))


def test_rearranged_wilson():
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
            stmt='dw_rearr(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson_r; dw_rearr = dirac_wilson_r(U,m)',
            globals={'U': U, 'v': v, 'm': mass},
            num_threads=tn
        )

        t1 = benchmark.Timer(
            stmt='dw_old(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson; dw_old = dirac_wilson(U,m)',
            globals={'U': U, 'v': v, 'm': mass},
            num_threads=tn
        )

        t2 = benchmark.Timer(
            stmt='dw_re2(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson_r2; dw_re2 = dirac_wilson_r2(U,m)',
            globals={'U': U, 'v': v, 'm': mass},
            num_threads=tn
        )

        # note: only shown when enabling stdout in pytest via -s argument
        print(t0.timeit(20+20*tn))
        print(t1.timeit(20+20*tn))
        print(t2.timeit(20+20*tn))

    print("=========================\n")

    dw_rearr = qcd_ml_accel_dirac.dirac_wilson_r(U,mass)
    dw_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)
    dw_re2 = qcd_ml_accel_dirac.dirac_wilson_r2(U,mass)

    assert all([torch.allclose(dw_rearr(v),dw_cpp(v)), torch.allclose(dw_cpp(v),dw_re2(v))])


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
        print(t1.timeit(20+20*tn))

    print("=========================\n")

    dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(U,mass,csw)
    dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)

    assert torch.allclose(dwc(v),dwc_cpp(v))


def test_rearranged_wilson_clover():
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
            stmt='dwc_rearr(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson_clover_r; dwc_rearr = dirac_wilson_clover_r(U,m,c)',
            globals={'U': U, 'v': v, 'm': mass, 'c': csw},
            num_threads=tn
        )

        t1 = benchmark.Timer(
            stmt='dwc_old(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson_clover; dwc_old = dirac_wilson_clover(U,m,c)',
            globals={'U': U, 'v': v, 'm': mass, 'c': csw},
            num_threads=tn
        )

        t2 = benchmark.Timer(
            stmt='dwc_re2(v)',
            setup='from qcd_ml_accel_dirac import dirac_wilson_clover_r2; dwc_re2 = dirac_wilson_clover_r2(U,m,c)',
            globals={'U': U, 'v': v, 'm': mass, 'c': csw},
            num_threads=tn
        )

        # note: only shown when enabling stdout in pytest via -s argument
        print(t0.timeit(20+20*tn))
        print(t1.timeit(20+20*tn))
        print(t2.timeit(20+20*tn))

    print("=========================\n")

    dwc = qcd_ml_accel_dirac.dirac_wilson_clover_r(U,mass,csw)
    dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U,mass,csw)
    dwcr2 = qcd_ml_accel_dirac.dirac_wilson_clover_r2(U,mass,csw)

    assert all([torch.allclose(dwc(v),dwc_cpp(v)), torch.allclose(dwc_cpp(v),dwcr2(v))])
