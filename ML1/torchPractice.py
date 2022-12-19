import numpy as np
import torch

x = torch.rand(3,2)
y = torch.zeros_like(x)
print(y)

x = torch.ones(2,3,1)
y = torch.ones_like(x)
print(y)

# numpy 2 tensor (这种方法生成的,numpy和tensor共享一个内存)
arr_np = np.random.rand(3,4)
y = torch.from_numpy(arr_np)
print(y)
arr_np[0,0] = 100
print(y)

# tensor 2 numpy
x = y.numpy()
print(x)


