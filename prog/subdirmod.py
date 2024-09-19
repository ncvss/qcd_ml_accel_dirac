import torch
import qcd_ml_accel_dirac

t1 = torch.tensor([[[[[[1,0,0],[0,1,0],[0,0,1]]]]]], dtype=torch.cdouble)
t2 = torch.tensor([[[[[[1j,1j,1j],[0,1,0],[0,0,1]]]]]], dtype=torch.cdouble)

print(torch.ops.qcd_ml_accel_dirac.shift_gaugemul(t1,t2,[0,0,0,0],[0,0,0,0]))
