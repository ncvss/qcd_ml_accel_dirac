import torch
from . import _C, grads
from .dirac_class import (
    dirac_wilson, dirac_wilson_clover, domain_wall_dirac, dirac_wilson_avx, dirac_wilson_clover_avx, dirac_wilson_clover_avx_old
)
from .plaq_p import _plaq_action_py, plaquette_action
#from .compat import lattice_to_array, array_to_lattice, array_to_lattice_list

