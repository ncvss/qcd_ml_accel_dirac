import torch

x = torch.tensor([[[1+1j,8],[2,3]],[[4,8j],[1,1]]],dtype=torch.cdouble)

print(x)
# gibt nur den Realteil zurück
print(x.type(torch.double))

# wandelt wie erwünscht in float64 um
print(torch.real(x))
print(torch.imag(x))

print(torch.stack((torch.real(x),torch.imag(x),),dim=-1))
