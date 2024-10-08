import torch
import torch.utils.benchmark as benchmark
import socket

import qcd_ml_accel_dirac

# # functions to compute plaquette in python
# def plaquette(U,mu,nu):
#     return torch.matmul(
#         U[mu], torch.matmul(
#             torch.roll(U[nu],-1,mu), torch.matmul(
#                 torch.adjoint(torch.roll(U[mu],-1,nu)), torch.adjoint(U[nu])
#                 )
#             )
#         )

# def plaq_action_py(U,g):
#     result = 0.0
#     for nu in range(4):
#         for mu in range(nu):
#             Umn = plaquette(U,mu,nu)
#             # trace( identity - Umn(x) ) = 3 - Umn(x)[0,0] - Umn(x)[1,1] - Umn(x)[2,2]
#             # we slice Umn to make the calculation at every point and then sum over all elements
#             result += torch.sum(torch.real(-Umn[...,0,0] - Umn[...,1,1] - Umn[...,2,2] + 3.0))
    
#     return 2.0/g**2 * float(result)


def test_plaquette_action_bench():



    num_threads = torch.get_num_threads()
    print("\n=======Test output=======")
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    size = [8,8,8,16]

    U = torch.randn([4,8,8,8,16,3,3],dtype=torch.cdouble)
    g = 2.0

    for tn in range(1,num_threads+1):

        t0 = benchmark.Timer(
            stmt='plaq_action_py(U,g)',
            setup='from qcd_ml_accel_dirac import plaq_action_py',
            globals={'U': U, 'g': g},
            num_threads=tn
        )

        t1 = benchmark.Timer(
            stmt='torch.ops.qcd_ml_accel_dirac.plaquette_action(U,g)',
            setup='import qcd_ml_accel_dirac',
            globals={'U': U, 'g': g},
            num_threads=tn
        )

        # note: only shown when enabling stdout in pytest via -s argument
        print(t0.timeit(200))
        print(t1.timeit(200))

    print("=========================\n")

    plaq_py = qcd_ml_accel_dirac.plaq_action_py(U,g)
    plaq_cpp = torch.ops.qcd_ml_accel_dirac.plaquette_action(U,g)
    print(plaq_py)
    print(plaq_cpp)

    assert abs(plaq_py-plaq_cpp) < 0.0001 * abs(plaq_py)
    #assert torch.allclose(plaq_cpp,plaq_py)
