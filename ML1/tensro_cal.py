import numpy as np
import torch

x = torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float32)
y = torch.tensor([[1,1],[2,2],[3,3]],dtype=torch.float32)

# method 1
print(f"method 1: {x+y}")

# method 2
print(f"method 2: {torch.add(x,y)}")

# method 3
x.add_(y) # 把y加到x的上面
print(f"method 3: {x}")
print('\n')

# -------------- 减法 -------------- #

# method 1
print(f"method 1: {x-y}")

# method 2
print(f"method 2: {torch.sub(x,y)}")

# method 3
x.sub_(y) # 把y加到x的上面
print(f"method 3: {x}")
print('\n')

# -------------- 乘法 -------------- #

# method 1
print(f"method 1: {x*y}")  # /

# method 2
print(f"method 2: {torch.mul(x,y)}")  # div

# method 3
x.mul_(y) # div
print(f"method 3: {x}")

# -------------- 归并运算 -------------- #
x = torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float32)
# 1. 所有元素的和
print(f"所有元素的和:{x.sum()}")
print(f"每一行的和:{x.sum(axis=1)}") # (3,2) --> (3,1)
print(f"每一列的和:{x.sum(axis=0)}") # (3,2) --> (1,2)
print(f"每一行的均值:{x.mean(axis=1)}") # (3,2) --> (3,1)
print(f"每一列的均值:{x.mean(axis=0)}") # (3,2) --> (1,2)

# -------------- 矩阵运算 -------------- #
y = torch.tensor([[1,1],[2,2],[3,3]],dtype=torch.float32)
print(f"x.T:{x.T}")
print(f"torch.matmul:{torch.matmul(x.T,y)}") # (2,3) * (3,2) --> (2,2)

# -------------- 索引 -------------- #
torch.manual_seed(0)
x = torch.rand(4,5)
print(f"原始数组:{x}")
print(f"x[0,0]:{x[0,0]}")
print(f"x[0,:]:{x[0,:]}")
print(f"x[:,0]:{x[:,0]}")

# -------------- 改变数组的大小 -------------- #
x = torch.rand(20)
print(f"原始数组:{x}, x data prt:{x.data_ptr()}") # 返回tensor首元素的内存地址
# method 1
y = x.view(4,5)
print(f"x.view{y}, y data prt:{y.data_ptr()}") # 返回tensor首元素的内存地址
# method 2
y = x.reshape(4,5)
print(f"x.reshape{y}, y data prt:{y.data_ptr()}")

xt = y.T #(5,4)
# z = xt.view(1,20)
z1 = xt.contiguous().view(1,20)
print(f"after view{z1}, data ptr:{z1.data_ptr()}")     # 对于不连续的数组,又分配了一个内存空间
z2 = xt.reshape(1,20)
print(f"after reshape{z2}, data ptr:{z2.data_ptr()}")  # 对于不连续的数组,又分配了一个内存空间

# method 3
y = x.unsqueeze(0) # 在第一个维度上增加一个维度
print(f"after unsqueeze:{y},shape:{y.shape}")
y = x.squeeze(0)  # 在第一个维度上去掉一个维度
print(f"after squeeze:{y},shape:{y.shape}")