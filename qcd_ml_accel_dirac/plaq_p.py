import torch


# try the plaquette only with torch functions

def plaquette(U,mu,nu):
    return torch.matmul(
        U[mu], torch.matmul(
            torch.roll(U[nu],-1,mu), torch.matmul(
                torch.adjoint(torch.roll(U[mu],-1,nu)), torch.adjoint(U[nu])
                )
            )
        )

def plaq_action_py(U,g):
    result = 0.0
    for nu in range(4):
        for mu in range(nu):
            Umn = plaquette(U,mu,nu)
            # trace( identity - Umn(x) ) = 3 - Umn(x)[0,0] - Umn(x)[1,1] - Umn(x)[2,2]
            # we slice Umn to make the calculation at every point and then sum over all elements
            result += torch.sum(torch.real(-Umn[...,0,0] - Umn[...,1,1] - Umn[...,2,2] + 3.0))
    
    return 2.0/g**2 * result
