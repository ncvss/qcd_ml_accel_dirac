# qcd_ml_accel_dirac

Designed as an extension for [qcd_ml](https://github.com/daknuett/qcd_ml). Operates on pytorch tensors
that are set up like in qcd_ml.

Provides two classes `dirac_wilson` and `dirac_wilson_clover` that define the corresponding operator
for a specific gauge configuration, mass and coupling. They are designed to be used interchangeably with
those from qcd_ml.

The ``compat`` submodule contains functions to convert between pytorch tensors and
[gpt](https://github.com/lehner/gpt) lattices.


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
The benchmark tests create custom output, so the ``-s`` option is required for them.
