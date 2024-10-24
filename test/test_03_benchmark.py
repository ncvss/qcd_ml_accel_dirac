import torch
import torch.utils.benchmark as benchmark
import socket

import qcd_ml_accel_dirac

# test theoretical performance of the computer with some simple computations

def test_max_flops_with_matmul():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
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
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

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


def test_memory_throughput_with_muladd():
    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    #num_threads = 1

    a = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)
    b = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)
    c = torch.randn([8,8,8,16,4,4], dtype = torch.cdouble)

    t0 = benchmark.Timer(
        stmt = 'torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar(a,b,c)',
        setup = 'import qcd_ml_accel_dirac',
        globals = {'a': a, 'b': b, 'c': c},
        num_threads = num_threads
    )
    t1 = benchmark.Timer(
        stmt = 'torch.ops.qcd_ml_accel_dirac.muladd_bench_par(a,b,c)',
        setup = 'import qcd_ml_accel_dirac',
        globals = {'a': a, 'b': b, 'c': c},
        num_threads = num_threads
    )
    t2 = benchmark.Timer(
        stmt = 'qcd_ml_accel_dirac.muladd_bench_nopar_py(a,b,c)',
        setup = 'import qcd_ml_accel_dirac',
        globals = {'a': a, 'b': b, 'c': c},
        num_threads = num_threads
    )
    t3 = benchmark.Timer(
        stmt = 'qcd_ml_accel_dirac.muladd_bench_par_py(a,b,c)',
        setup = 'import qcd_ml_accel_dirac',
        globals = {'a': a, 'b': b, 'c': c},
        num_threads = num_threads
    )

    print(t0.timeit(30000))
    print(t1.timeit(30000))
    print(t2.timeit(30000))
    print(t3.timeit(30000))

    res_py = qcd_ml_accel_dirac.muladd_bench_nopar_py(a,b,c)
    res_cpp = torch.ops.qcd_ml_accel_dirac.muladd_bench_nopar(a,b,c)

    assert torch.allclose(res_py,res_cpp)



