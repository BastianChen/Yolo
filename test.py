import torch
import numpy as np

a = torch.tensor([1, 0, -0.8, 0.99])
b = torch.sigmoid_(a)
a = torch.tensor([1, 0, -0.8, 0.99])
c = torch.cat((a, b))
print(b)
print(c)

# a = np.array([2, 5, 0.2, 3])
# index = [0, 2]
# b = np.delete(a, index)
# print(b)
