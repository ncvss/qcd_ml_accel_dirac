import torch
import qcd_ml_accel_dirac

U = torch.randn([4,4,4,4,8,3,3], dtype=torch.cdouble)
v = torch.randn([4,4,4,8,4,3], dtype=torch.cdouble)
m = -0.5

dw = qcd_ml_accel_dirac.dirac_wilson(U,m)

vp = dw(v)

print(vp[0,0,0,0])
