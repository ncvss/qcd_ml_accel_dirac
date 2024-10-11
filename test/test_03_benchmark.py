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

    dwc = qcd_ml.qcd.dirac.dirac_wilson(U,mass)
    dwc_cpp = qcd_ml_accel_dirac.dirac_wilson(U,mass)

    assert torch.allclose(dwc(v),dwc_cpp(v))


def test_max_flops_with_matmul():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print(f'Machine has {num_threads} threads')

    # use matrix multiplication as a test for the theoretical performance

    m1 = torch.randn([12*16,87],dtype=torch.cdouble)
    m2 = torch.randn([87,8*8*8],dtype=torch.cdouble)

    for tn in range(1,num_threads+1):
        t0 = benchmark.Timer(
            stmt='torch.matmul(m1,m2)',
            globals={'m1': m1, 'm2': m2},
            num_threads=tn
        )
        print(t0.timeit(1000))
    
    print("=========================\n")

    assert True

def test_max_flops_with_muladd():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")

    # use a[i] * b[i] + c as a test for the theoretical performance

    a = torch.randn([2076 * 8**3 * 16],dtype=torch.cdouble)
    b = torch.randn([2076 * 8**3 * 16],dtype=torch.cdouble)

    for tn in range(1,num_threads+1):
        t0 = benchmark.Timer(
            # takes too long to be usable as test
            # stmt='for i in range(2076 * 8**3 * 16): a[i] * b[i] + 1.2',
            stmt='torch.mul(a,b)+1.2',
            globals={'a': a, 'b': b},
            num_threads=tn
        )
        print(t0.timeit(30))
    
    print("=========================\n")

    assert True

