import numpy as np
import torch
#import qcd_ml_accel_dirac

import gpt as g
from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice

rng = g.random("test")
U = g.qcd.gauge.random(g.grid([2,4,8,16], g.double), rng)
grid = U[0].grid
m = 0.08
m5 = 1.8

mobius_params = {
    "mass": m,
    "M5": m5,
    "b": 0.0, #1.5?
    "c": 0.0, #0.5?
    "Ls": 32,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}

qm = g.qcd.fermion.mobius(U, mobius_params)
grid5 = qm.F_grid
src = g.vspincolor(grid5)
rng.cnormal(src)

def lattice2nd_2(lattice):
    shape = lattice.grid.fdimensions
    print("gpt shape:",shape)
    shape = list(reversed(shape))
    if lattice[:].shape[1:] != (1,):
        shape.extend(lattice[:].shape[1:])
    #print(shape)
    print("underlying np shape in gpt",lattice[:].shape)
    # Wir haben durch Tests herausgefunden, dass die Daten im Speicher falsch herum liegen
    result = lattice[:].reshape(shape)
    print("result np shape:",result.shape)
    result = np.swapaxes(result, 0, 3)
    result = np.swapaxes(result, 1, 2)
    print("result np shape after swap:",result.shape)
    return result

print("U[0]")
lattice2nd_2(U[0])
print("vector field")
lattice2nd_2(src)

# lattice[:] ist ein np-Tensor, der als ersten Index die Gitter-Indizes geplättet
# und in verkehrter Reihenfolge hat, die nächsten Indizes sind Spin und/oder Gauge

# for i in range(5):
#     print(src[tuple(1 if n==i else 0 for n in range(5))])
#     print(src[:][8**i])
# # Diese Tests zeigen noch einmal, dass die Daten in gpt so liegen,
# # dass der erste Index der am schnellsten laufende ist.

t = np.array([1,2,3.0])
print(t.shape)
print(t.shape[1:] == (1,))

    
# convert a lattice to a numpy array
def pure_lattice_to_array(latt):
    grid_shape = latt.grid.fdimensions
    grid_shape = list(reversed(grid_shape))
    dof_shape = latt[:].shape[1:]
    if dof_shape == (1,):
        dof_shape = []
    else:
        dof_shape = list(dof_shape)
    
    ndims = len(grid_shape)
    ndofs = len(dof_shape)
    order = list(reversed(range(ndims))) + list(range(ndims,ndims+ndofs))

    res = latt[:].reshape(grid_shape + dof_shape)
    res = np.transpose(res, order)
    return res

# wrapper that can also handle lists of lattices and turns them into numpy arrays
def lattice_to_array(lattice):
    if isinstance(lattice, list):
        result = [pure_lattice_to_array(l) for l in lattice]
        return np.stack(result, axis=0)
    else:
        return pure_lattice_to_array(lattice)

Ut = lattice_to_array(U)
st = lattice_to_array(src)
print("new function")
print("shape of U:", Ut.shape)
print("shape of vector:", st.shape)
