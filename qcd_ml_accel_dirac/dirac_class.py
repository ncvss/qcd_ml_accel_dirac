import torch
import numpy as np


# path buffer only for intermediate computations
# derived from the version in qcd_ml
class _PathBufferTemp:
    def __init__(self, U, path):
        self.path = path

        self.accumulated_U = torch.zeros_like(U[0])
        self.accumulated_U[:,:,:,:] = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.cdouble)

        for mu, nhops in self.path:
            if nhops < 0:
                direction = -1
                nhops *= -1
            else:
                direction = 1

            for _ in range(nhops):
                if direction == -1:
                    U = torch.roll(U, 1, mu + 1) # mu + 1 because U is (mu, x, y, z, t)
                    self.accumulated_U = torch.matmul(U[mu], self.accumulated_U)
                else:
                    self.accumulated_U = torch.matmul(U[mu].adjoint(), self.accumulated_U)
                    U = torch.roll(U, -1, mu + 1)

# gamma and sigma matrices for intermediate computations
_gamma = [torch.tensor([[0,0,0,1j]
                ,[0,0,1j,0]
                ,[0,-1j,0,0]
                ,[-1j,0,0,0]], dtype=torch.cdouble)
    , torch.tensor([[0,0,0,-1]
                ,[0,0,1,0]
                ,[0,1,0,0]
                ,[-1,0,0,0]], dtype=torch.cdouble)
    , torch.tensor([[0,0,1j,0]
                ,[0,0,0,-1j]
                ,[-1j,0,0,0]
                ,[0,1j,0,0]], dtype=torch.cdouble)
    , torch.tensor([[0,0,1,0]
                ,[0,0,0,1]
                ,[1,0,0,0]
                ,[0,1,0,0]], dtype=torch.cdouble)
    ]

_sigma = [[(torch.matmul(_gamma[mu], _gamma[nu]) 
            - torch.matmul(_gamma[nu], _gamma[mu])) / 2
            for nu in range(4)] for mu in range(4)]

# masks to choose upper and lower triangle
_triag_mask_1 = torch.tensor([[(sw < 6 and sh < 6 and sh <= sw) for sw in range(12)] for sh in range(12)],
                            dtype=torch.bool)
_triag_mask_2 = torch.tensor([[(sw >= 6 and sh >= 6 and sh <= sw) for sw in range(12)] for sh in range(12)],
                            dtype=torch.bool)


# Dirac Wilson operator, using C++ functions
class dirac_wilson:
    """
    Wilson Dirac operator with gauge config U.
    """
    def __init__(self, U: torch.Tensor, mass_parameter: float):

        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        """
        Gauge configuration stored as a complex tensor of shape (4,Lx,Ly,Lz,Lt,3,3).

        At each site, there are 4 gauge links (for each direction), which are SU(3) matrices.
        """

        self.mass_parameter = mass_parameter
        """Mass parameter"""

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.dirac_wilson_call(self.U, v, self.mass_parameter)


class dirac_wilson_avx:
    """
    Wilson Dirac operator with gauge config U
    that creates a lookup table for the hops and uses AVX instructions.

    The axes are U[mu,x,y,z,t,g,gi] and v[x,y,z,t,s,g].
    """
    def __init__(self, U: torch.Tensor, mass_parameter: float, boundary_phases: list=[1,1,1,1]):

        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        """
        Gauge configuration stored as a complex tensor of shape (4,Lx,Ly,Lz,Lt,3,3).

        At each site, there are 4 gauge links (for each direction), which are SU(3) matrices.
        """
        
        self.mass_parameter = mass_parameter
        """Mass parameter"""

        grid = U.shape[1:5]
        strides = torch.tensor([grid[1]*grid[2]*grid[3], grid[2]*grid[3], grid[3], 1], dtype=torch.int32)
        npind = np.indices(grid, sparse=False)
        indices = torch.tensor(npind, dtype=torch.int32).permute((1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)

        hop_inds = []
        for coord in range(4):
            # index after a negative step in coord direction
            minus_hop_ind = torch.clone(indices)
            minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]-1+grid[coord], grid[coord])
            # index after a positive step in coord direction
            plus_hop_ind = torch.clone(indices)
            plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+1, grid[coord])
            # compute flattened index by dot product with strides
            hop_inds.append(torch.matmul(minus_hop_ind, strides))
            hop_inds.append(torch.matmul(plus_hop_ind, strides))

        self.hop_inds = torch.stack(hop_inds, dim=1).contiguous()
        """
        For each lattice site, hop_inds contains the address of all shifts from that lattice site
        backward and forward in all coordinate directions.
        """

        hop_phases = torch.ones(list(grid)+[8], dtype=torch.int8)
        # for the sites at the lower boundary [0], a hop in negative direction has the phase
        # for the sites at the upper boundary [-1], a hop in positive direction has the phase
        for edge in range(2):
            hop_phases[-edge,:,:,:,0+edge] = boundary_phases[0]
            hop_phases[:,-edge,:,:,2+edge] = boundary_phases[1]
            hop_phases[:,:,-edge,:,4+edge] = boundary_phases[2]
            hop_phases[:,:,:,-edge,6+edge] = boundary_phases[3]
        self.hop_phases = hop_phases
        """
        For each lattice site, hop_phases contains the phase factor for all shifts from that lattice site
        backward and forward in all coordinate directions.
        """

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.dw_avx_templ(self.U, v, self.hop_inds, self.hop_phases,
                                                         self.mass_parameter)
    

# Dirac Wilson operator with clover term improvement, using C++
class dirac_wilson_clover:
    """
    Wilson clover Dirac operator with gauge config U.
    """
    def __init__(self, U: torch.Tensor, mass_parameter: float, csw: float):

        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        """
        Gauge configuration stored as a complex tensor of shape (4,Lx,Ly,Lz,Lt,3,3).

        At each site, there are 4 gauge links (for each direction), which are SU(3) matrices.
        """

        self.mass_parameter = mass_parameter
        """Mass parameter"""

        self.csw = csw
        """Sheikholeslami-Wohlert coefficient"""

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U
        
        # only a flat list, as it needs to be accessed by C++
        self.field_strength = []
        """
        List of tensors of shape (Lx,Ly,Lz,Lt,3,3) with the field strength matrix at every site.

        The list corresponds to the coordinate directions (mu,nu) = (0,0), (1,0), (2,0), (2,1), (3,1), (3,2)
        """

        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                self.field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        
    def __call__ (self, v):
        return torch.ops.qcd_ml_accel_dirac.dirac_wilson_clover_call(self.U, v, self.field_strength,
                                                                     self.mass_parameter, self.csw)



class dirac_wilson_clover_avx_old:
    """
    Wilson clover Dirac operator with gauge config U
    that creates a lookup table for the hops and uses AVX instructions.

    The axes are U[mu,x,y,z,t,g,gi], v[x,y,z,t,s,gi] and F[x,y,z,t,munu,g,gi].

    The field strength tensor is precomputed naively.
    """
    def __init__(self, U: torch.Tensor, mass_parameter: float, csw: float):
        
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        """
        Gauge configuration stored as a complex tensor of shape (4,Lx,Ly,Lz,Lt,3,3).

        At each site, there are 4 gauge links (for each direction), which are SU(3) matrices.
        """

        self.mass_parameter = mass_parameter
        """Mass parameter"""

        self.csw = csw
        """Sheikholeslami-Wohlert coefficient"""

        grid = U.shape[1:5]
        strides = torch.tensor([grid[1]*grid[2]*grid[3], grid[2]*grid[3], grid[3], 1], dtype=torch.int32)
        npind = np.indices(grid, sparse=False)
        indices = torch.tensor(npind, dtype=torch.int32).permute((1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)

        hop_inds = []
        for coord in range(4):
            # index after a negative step in coord direction
            minus_hop_ind = torch.clone(indices)
            minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]-1+grid[coord], grid[coord])
            # index after a positive step in coord direction
            plus_hop_ind = torch.clone(indices)
            plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+1, grid[coord])
            # compute flattened index by dot product with strides
            hop_inds.append(torch.matmul(minus_hop_ind, strides))
            hop_inds.append(torch.matmul(plus_hop_ind, strides))
        
        self.hop_inds = torch.stack(hop_inds, dim=1).contiguous()
        """
        For each lattice site, hop_inds contains the address of all shifts from that lattice site
        forward and backward in all coordinate directions.
        """

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U
        
        field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        
        self.field_strength = torch.stack(field_strength, dim=4)
        """
        Tensor of shape (Lx,Ly,Lz,Lt,6,3,3) with the field strength matrices
        for all combinations of coordinate directions at every site.

        The directions are ordered (mu,nu) = (0,0), (1,0), (2,0), (2,1), (3,1), (3,2)
        """

        assert tuple(self.field_strength.shape[4:7]) == (6,3,3,)

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.dwc_avx_templ(self.U, v, self.field_strength,
                                                          self.hop_inds, self.mass_parameter,
                                                          self.csw)

class dirac_wilson_clover_avx:
    """
    Wilson clover Dirac operator with gauge config U
    that creates a lookup table for the hops and uses AVX instructions.

    field_strength * sigma * v is precomputed by computing the tensor product
    of field_strength * sigma, and only the upper triangle of two 6x6 blocks
    is passed for the field strength.
    """
    def __init__(self, U: torch.Tensor, mass_parameter: float, csw: float, boundary_phases: list=[1,1,1,1]):
        
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        """
        Gauge configuration stored as a complex tensor of shape (4,Lx,Ly,Lz,Lt,3,3).

        At each site, there are 4 gauge links (for each direction), which are SU(3) matrices.
        """

        self.mass_parameter = mass_parameter
        """Mass parameter"""

        self.csw = csw
        """Sheikholeslami-Wohlert coefficient"""

        grid = [U.shape[1], U.shape[2], U.shape[3], U.shape[4]]
        strides = torch.tensor([grid[1]*grid[2]*grid[3], grid[2]*grid[3], grid[3], 1], dtype=torch.int32)
        npind = np.indices(grid, sparse=False)
        indices = torch.tensor(npind, dtype=torch.int32).permute((1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)

        hop_inds = []
        for coord in range(4):
            # index after a negative step in coord direction
            minus_hop_ind = torch.clone(indices)
            minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]-1+grid[coord], grid[coord])
            # index after a positive step in coord direction
            plus_hop_ind = torch.clone(indices)
            plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+1, grid[coord])
            # compute flattened index by dot product with strides
            hop_inds.append(torch.matmul(minus_hop_ind, strides))
            hop_inds.append(torch.matmul(plus_hop_ind, strides))

        self.hop_inds = torch.stack(hop_inds, dim=1).contiguous()
        """
        For each lattice site, hop_inds contains the address of all shifts from that lattice site
        forward and backward in all coordinate directions.
        """

        hop_phases = torch.ones(list(grid)+[8], dtype=torch.int8)
        # for the sites at the lower boundary [0], a hop in negative direction has the phase
        # for the sites at the upper boundary [-1], a hop in positive direction has the phase
        for edge in range(2):
            hop_phases[-edge,:,:,:,0+edge] = boundary_phases[0]
            hop_phases[:,-edge,:,:,2+edge] = boundary_phases[1]
            hop_phases[:,:,-edge,:,4+edge] = boundary_phases[2]
            hop_phases[:,:,:,-edge,6+edge] = boundary_phases[3]
        self.hop_phases = hop_phases
        """
        For each lattice site, hop_phases contains the phase factor for all shifts from that lattice site
        backward and forward in all coordinate directions.
        """
        
        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U

        dim = list(U.shape[1:5])
        self.dim = dim
        # tensor product of the sigma matrix and field strength tensor
        field_strength_sigma = torch.zeros(dim+[4,3,4,3], dtype=torch.cdouble)
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                Fmunu = (Qmunu[mu][nu] - Qmunu[nu][mu]) / 8
                Fsigma = torch.einsum('xyztgh,sr->xyztsgrh',Fmunu,_sigma[mu][nu])
                # csw gets absorbed into the matrices
                field_strength_sigma += 2*(-self.csw/4)*Fsigma
        
        field_strength_sigma = field_strength_sigma.contiguous().reshape(dim+[12,12])
        # this should be hermitian and have two 6x6 blocks (diagonal has numerical artifacts)

        self.field_strength_sigma = torch.stack([field_strength_sigma[:,:,:,:,_triag_mask_1],
                                                 field_strength_sigma[:,:,:,:,_triag_mask_2]],dim=-1)
        """
        Tensor that contains matrices needed to compute the clover term.

        This is precomputed by taking the Tensor product of sigma_{mu,nu}
        with the field strength matrix F_{mu,nu} at every lattice site,
        then summing over all combinations of (mu,nu) and multiplying the csw prefactor.
        The result is a tensor with shape (Lx,Ly,Lz,Lt,12,12).

        As these 12x12 matrices are hermitian and only have 2 nonzero 6x6 blocks,
        only the upper triangle of these 2 blocks is stored
        as a tensor with shape (Lx,Ly,Lz,Lt,21,2).
        """
        assert tuple(self.field_strength_sigma.shape[4:6]) == (21,2,)


    def __call__ (self, v):
        return torch.ops.qcd_ml_accel_dirac.dwc_avx_templ_grid(self.U, v, self.field_strength_sigma,
                                                               self.hop_inds, self.hop_phases, self.mass_parameter)



class domain_wall_dirac:
    """
    Domain Wall Dirac operator in Shamir formulation with gauge config U.
    """
    def __init__(self, U: torch.Tensor, mass_parameter: float, m5_parameter: float):

        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        """
        Gauge configuration stored as a complex tensor of shape (4,Lx,Ly,Lz,Lt,3,3).

        At each site, there are 4 gauge links (for each direction), which are SU(3) matrices.
        """

        self.mass_parameter = mass_parameter
        """Bare mass parameter"""

        self.m5_parameter = m5_parameter
        """Bulk mass parameter"""

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.domain_wall_dirac_call(self.U, v, self.mass_parameter, self.m5_parameter)

