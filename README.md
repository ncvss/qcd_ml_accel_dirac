# qcd_ml_accel_dirac

Designed as an extension for [qcd_ml](https://github.com/daknuett/qcd_ml).

Provides two classes `dirac_wilson` and `dirac_wilson_clover` that define the corresponding operator
for a specific gauge configuration, mass and coupling. See arXiv:2302.05419 for the definition of those
operators.

The operator can be applied on a vector field by calling the class instance.
