import torch

def evenmask(dims):
    return torch.tensor([[[(y+z+t)%2 == 0 for t in range(dims[2])] for z in range(dims[1])]
      for y in range(dims[0])], dtype=torch.bool)

lat_dim = [4,4,4]

v = torch.randn(lat_dim, dtype=torch.double)
U = torch.randn([2] + lat_dim, dtype=torch.double)
m = evenmask(lat_dim)

eo_dim = lat_dim[:]
eo_dim[-1] //= 2

print("lat dim:", lat_dim)
print("dim of even/odd tensors:", eo_dim)

print("mask:")
print(m)

print("v:")
print(v)

print("v even:")
print(v[m])
print("in correct shape:")
print(v[m].reshape(eo_dim))

print("v odd, reshaped:")
print(v.roll(-1,2)[m].reshape(eo_dim))

print("U:")
print(U)

print("U even:")
print(U[:,m])
print("in correct shape:")
print(U[:,m].reshape([2]+eo_dim))

# scheint alles zu funktionieren

vbig = torch.randn(lat_dim+[2], dtype=torch.double)
print("vbig:")
print(vbig)
print("vbig even:")
print(vbig[m])

# auch das lässt die unteren Dimensionen unberührt

