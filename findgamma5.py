import torch
from qcd_ml.qcd.static import gamma

# for gi in gamma:
#     print(gi)

# gamma5 = torch.matmul(gamma[3],torch.matmul(gamma[0],torch.matmul(gamma[1],gamma[2])))

gamma5 = torch.tensor([[ 1, 0, 0, 0],
                       [ 0, 1, 0, 0],
                       [ 0, 0,-1, 0],
                       [ 0, 0, 0,-1]], dtype=torch.cdouble)

print("gamma5",gamma5)

Pp = (torch.eye(4)+gamma5)/2
Pm = (torch.eye(4)-gamma5)/2

print(Pp)
print(Pm)

# Unfortunately, I do not know which representation is used for the gamma matrices,
# so I cannot find the projector.
# If gamma_mu are the gamma matrices in chiral representation, then:
# gamma[0] = i gamma_1
# gamma[1] = -i gamma_2
# gamma[2] = i gamma_3
# gamma[3] = gamma_0
# This is almost the euclidean chiral representation, except that gamma[1] has the wrong sign

# From https://github.com/lehner/gpt/blob/master/lib/gpt/core/gamma.py I took the definition:
# gamma5 = [[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]

# This means:
# Pplus = [[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
# Pminus = [[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]
