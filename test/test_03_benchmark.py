import torch
import torch.utils.benchmark as benchmark
import socket


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

