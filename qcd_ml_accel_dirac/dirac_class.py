import torch


# path buffer only for intermediate computations
# derived from the version in qcd_ml
class _PathBufferTemp:
    def __init__(self, U, path):
        if isinstance(U, list):
            # required by torch.roll below.
            U = torch.stack(U)
        self.path = path

        self.accumulated_U = torch.zeros_like(U[0])
        self.accumulated_U[:,:,:,:] = torch.complex(
                torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.double)
                , torch.zeros(3, 3, dtype=torch.double)
                )

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


# Dirac Wilson operator, using C++ functions
class dirac_wilson:
    """
    Dirac Wilson operator with gauge config U.
    """
    def __init__(self, U, mass_parameter):
        if isinstance(U, list):
            U = torch.stack(U)
        self.U = U
        self.mass_parameter = mass_parameter

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.dirac_wilson_call(self.U, v, self.mass_parameter)
    

        
# Dirac Wilson operator with clover term improvement, using C++
class dirac_wilson_clover:
    """
    Dirac Wilson Clover operator with gauge config U.
    """
    def __init__(self, U, mass_parameter, csw):
        if isinstance(U, list):
            U = torch.stack(U)
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

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
                        Qmunu[mu][nu] += plaquette_path_buffers[mu][nu][ii].accumulated_U
        
        # only a flat list, as it needs to be accessed by C++
        self.field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                self.field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        
    def __call__ (self, v):
        return torch.ops.qcd_ml_accel_dirac.dirac_wilson_clover_call(self.U, v, self.field_strength, self.mass_parameter, self.csw)


class domain_wall_dirac:
    """
    Domain Wall Dirac operator in Shamir formulation with gauge config U.
    """
    def __init__(self, U, mass_parameter, m5_parameter):
        self.U = U
        self.mass_parameter = mass_parameter
        self.m5_parameter = m5_parameter

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.domain_wall_dirac_call(self.U, v, self.mass_parameter, self.m5_parameter)

