import numpy as np
import gpt as g

# convert a lattice of arbitrary dimensions to a numpy array
def pure_lattice_to_array(lattice):
    # length of the space-time axes in the in-memory order
    grid_shape = lattice.grid.fdimensions
    grid_shape = list(reversed(grid_shape))

    # spin and gauge degrees of freedom
    dof_shape = lattice[:].shape[1:]
    if dof_shape == (1,):
        dof_shape = []
    else:
        dof_shape = list(dof_shape)
    
    # order of the indidices that we want in the numpy array
    ndims = len(grid_shape)
    ndofs = len(dof_shape)
    order = list(reversed(range(ndims))) + list(range(ndims, ndims + ndofs))

    result = lattice[:].reshape(grid_shape + dof_shape)
    result = np.transpose(result, order)
    return result

# wrapper that can also handle lists of lattices and turns them into numpy arrays
def lattice_to_array(lattice):
    if isinstance(lattice, list):
        result = [pure_lattice_to_array(l) for l in lattice]
        return np.stack(result, axis=0)
    else:
        return pure_lattice_to_array(lattice)
