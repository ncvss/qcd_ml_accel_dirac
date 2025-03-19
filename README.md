# qcd_ml_accel_dirac

Designed as an extension for [qcd_ml](https://github.com/daknuett/qcd_ml). Operates on gauge and vector fields
that are stored as Pytorch tensors, with the same memory setup as in qcd_ml.

## Functionality

The classes `dirac_wilson` and `dirac_wilson_clover` define the corresponding operator
for a specific gauge configuration, mass and coupling.
They are designed to be used interchangeably with those from qcd_ml.

The classes `dirac_wilson_avx` and `dirac_wilson_clover_avx` provide AVX vectorised
computations of those operators,
which can be called on AVX capable x86 computers.

The class `domain_wall_dirac` defines a Domain Wall Dirac operator in Shamir formulation.

Calling these classes with a vector field argument returns the field after applying the operator.

The `compat` submodule contains functions to convert between Pytorch tensors and
[gpt](https://github.com/lehner/gpt) lattices.

`plaquette_action` computes the plaquette action of a gauge configuration.


## Installation

To install, run in the repository directory:
````
pip install . --no-build-isolation
````

## Requirements

This package requires pytorch for CPU. The CPU specific installation of Pytorch for a specified
version is detailed [here](https://pytorch.org/get-started/previous-versions/).

The ``compat`` module also requires gpt and numpy.

qcd_ml is required to test against it.

## Tests

The tests can be run with pytest.
````
pytest -v -s test/
````
The benchmark tests create custom output, so the `-s` option is required for them.
