import torch
import os
import numpy as np

# try the plaquette only with torch functions

def plaquette(U,mu,nu):
    return torch.matmul(
        U[mu], torch.matmul(
            torch.roll(U[nu],-1,mu), torch.matmul(
                torch.adjoint(torch.roll(U[mu],-1,nu)), torch.adjoint(U[nu])
                )
            )
        )

def plaq_action(U,g):
    result = 0.0
    #vol = U.size(0)*U.size(1)*U.size(2)*U.size(3)
    for nu in range(4):
        for mu in range(nu):
            Umn = plaquette(U,mu,nu)
            # trace( identity - Umn(x) ) = 3 - Umn(x)[0,0] - Umn(x)[1,1] - Umn(x)[2,2]
            # we slice Umn to make the calculation at every point and then sum over all elements
            result += torch.sum(torch.real(-Umn[...,0,0] - Umn[...,1,1] - Umn[...,2,2] + 3.0))
    
    return 2.0/g**2 * result


# tensor that looks like a gauge field (one for each direction)
U = torch.randn([4,8,8,8,16,3,3])

base_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_1500 = torch.tensor(np.load(os.path.join(base_dir_path,"test","assets","1500.config.npy")))

print(U[...,0,0].size())

pl = plaquette(config_1500,0,2)
print(pl.shape)
print(pl[5,2,7,8])
print(pl[0,4,4,6])

print(plaq_action(config_1500,2.0)/(8**3*16)/6)
