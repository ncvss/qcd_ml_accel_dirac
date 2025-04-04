import numpy as np
import gpt as g # type: ignore
import torch


def _pure_lattice_to_array(lattice):
    """
    Convert a gpt lattice of arbitrary dimensions to a numpy.ndarray.
    """
    # length of the space-time axes in the in-memory order
    grid_shape = lattice.grid.fdimensions
    grid_shape = list(reversed(grid_shape))

    # spin and gauge degrees of freedom
    dof_shape = lattice[:].shape[1:]
    if dof_shape == (1,):
        dof_shape = []
    else:
        dof_shape = list(dof_shape)
    
    # order of the indices that we want in the numpy array
    ndims = len(grid_shape)
    ndofs = len(dof_shape)
    order = list(reversed(range(ndims))) + list(range(ndims, ndims + ndofs))

    result = lattice[:].reshape(grid_shape + dof_shape)
    result = np.transpose(result, order)
    return np.ascontiguousarray(result)

# wrapper that can also handle lists of lattices
def lattice_to_array(lattice):
    """
    Convert a gpt lattice or a list of lattices to a single numpy.ndarray.
    """
    if isinstance(lattice, list):
        result = [_pure_lattice_to_array(l) for l in lattice]
        return np.stack(result, axis=0)
    else:
        return _pure_lattice_to_array(lattice)


def array_to_lattice(arr, grid, lat_constructor):
    """
    Convert a numpy.ndarray or torch.Tensor to a gpt lattice on the given grid, using the given constructor.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy(force=True)

    lat = lat_constructor(grid)

    # order of indices in the gpt memory format
    ndims = len(grid.fdimensions)
    ndofs = arr.ndim - ndims
    order = list(reversed(range(ndims))) + list(range(ndims, ndims + ndofs))

    data = np.transpose(arr, order)

    # number of grid points
    vol = 1
    for i in range(ndims):
        vol *= data.shape[i]
    
    lat[:] = np.reshape(data, [vol] + list(data.shape[ndims:]))
    return lat

def array_to_lattice_list(arr, grid, lat_constructor):
    """
    Convert a numpy.ndarray, torch.Tensor or a list of the former to a list of gpt lattices.
    Mainly to be used for gauge fields that are stored as a single tensor.
    """
    res = []
    for subarr in arr:
        res.append(array_to_lattice(subarr, grid, lat_constructor))
    return res

