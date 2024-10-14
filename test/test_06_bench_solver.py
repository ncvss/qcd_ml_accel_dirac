import torch
import torch.utils.benchmark as benchmark
import socket

import qcd_ml_accel_dirac
import qcd_ml


def test_solver_with_python_vs_cpp_wilson():
    num_threads = torch.get_num_threads()
    print("running on host", socket.gethostname())
    print(f'Machine has {num_threads} threads')

    #U = torch.randn([4,8,8,8,16,3,3], dtype=torch.cdouble)

    #v = torch.randn([8,8,8,16,4,3], dtype=torch.cdouble)

    t0 = benchmark.Timer(
        stmt='qcd_ml.util.solver.GMRES(dwc_py, v, v, eps=1e-4, inner_iter=25, maxiter=400)',
        setup='import torch; import qcd_ml; '
            + 'U = torch.randn([4,8,8,8,16,3,3], dtype=torch.cdouble); '
            + 'v = torch.randn([8,8,8,16,4,3], dtype=torch.cdouble); '
            + 'dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U, -0.61, 1)',
        #globals={'U': U, 'v': v},
        num_threads=num_threads
    )
    t1 = benchmark.Timer(
        stmt='qcd_ml.util.solver.GMRES(dwc_cpp, v, v, eps=1e-4, inner_iter=25, maxiter=400)',
        setup='import torch; import qcd_ml; import qcd_ml_accel_dirac; '
            + 'U = torch.randn([4,8,8,8,16,3,3], dtype=torch.cdouble); '
            + 'v = torch.randn([8,8,8,16,4,3], dtype=torch.cdouble); '
            + 'dwc_cpp = qcd_ml_accel_dirac.dirac_wilson_clover(U, -0.61, 1)',
        #globals={'U': U, 'v': v},
        num_threads=num_threads
    )

    print(t0.timeit(3))
    print(t1.timeit(3))

    assert True

