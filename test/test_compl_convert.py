import torch
import qcd_ml_accel_dirac

x = torch.tensor([1+2j, 222+1.1j], dtype=torch.cdouble)
y = torch.ops.qcd_ml_accel_dirac.convert_complex_to_double(x)
print(y)
