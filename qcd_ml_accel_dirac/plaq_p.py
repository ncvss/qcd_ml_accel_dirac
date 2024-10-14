import torch

# plaquette action rename
# however, calling this is extremely slow
def plaquette_action(U,g):
    return torch.ops.qcd_ml_accel_dirac.plaquette_action(U,g)

# try the plaquette only with torch functions

def _plaquette(U,mu,nu):
    return torch.matmul(
        U[mu], torch.matmul(
            torch.roll(U[nu],-1,mu), torch.matmul(
                torch.adjoint(torch.roll(U[mu],-1,nu)), torch.adjoint(U[nu])
                )
            )
        )

def _plaq_action_py(U,g):
    result = 0.0
    for nu in range(4):
        for mu in range(nu):
            Umn = _plaquette(U,mu,nu)
            # trace( identity - Umn(x) ) = 3 - Umn(x)[0,0] - Umn(x)[1,1] - Umn(x)[2,2]
            # we slice Umn to make the calculation at every point and then sum over all elements
            result += torch.sum(torch.real(-Umn[...,0,0] - Umn[...,1,1] - Umn[...,2,2] + 3.0))
    
    return 2.0/g**2 * float(result) / (U.size(1)*U.size(2)*U.size(3)*U.size(4)*6)

