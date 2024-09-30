import torch

# gauge field (one for each direction)
U = torch.randn([4,4,4,4,8,3,3])

def plaquette(U,mu,nu):
    return torch.matmul(
        U[mu], torch.matmul(
            torch.roll(U[nu],-1,mu), torch.matmul(
                torch.conj(torch.roll(U[mu],-1,nu)), torch.conj(U[nu])
                )
            )
        )

def plaq_action(U,g):
    result = 0.0
    vol = U.size(0)*U.size(1)*U.size(2)*U.size(3)
    for nu in range(4):
        for mu in range(nu):
            Umn = plaquette(U,mu,nu)
            result += 2.0/g**2 * torch.real()