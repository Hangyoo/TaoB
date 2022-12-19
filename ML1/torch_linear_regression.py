import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)

'''手动写用numpy计算导数'''

x = np.array([[1,2],[2,3],[4,6],[3,1]],dtype=np.float32)
y = np.array([[8],[13],[26],[9]],dtype=np.float32)
# y = 2 * x1 + 3 * x2

w = np.random.rand(2,1)
iter_count = 300
lr = 0.02

def forward(x):
    return np.matmul(x,w) # (4.2) * (2*1) --> (4,1)

def loss(y,y_pred):
    return ((y-y_pred)**2 / 2).sum()

def gradient(x,y,y_pred):
    return np.matmul(x.T, y_pred - y)

# 更新参数过程
for i in range(iter_count):
    # forward
    y_pred = forward(x)
    l = loss(y,y_pred)
    print(f'iter {i}, loss {l}')

    # backward
    grad = gradient(x,y,y_pred)

    w -= lr * grad

print(f'final parameter:{w}') # 得到[2,3]

x1 = 4
x2 = 5
# 按公式 y = 2 * x1 + 3 * x2, 则y=23
print(forward(np.array([[x1,x2]])))


'''用pytorch计算上述过程'''
x = torch.tensor([[1,2],[2,3],[4,6],[3,1]],dtype=torch.float32)
y = torch.tensor([[8],[13],[26],[9]],dtype=torch.float32)
# y = 2 * x1 + 3 * x2

w = torch.rand(2,1,requires_grad=True,dtype=torch.float32)
iter_count = 300
lr = 0.02

def forward(x):
    return torch.matmul(x,w) # (4.2) * (2*1) --> (4,1)

def loss(y,y_pred):
    return ((y-y_pred)**2 / 2).sum()

# def gradient(x,y,y_pred): torch 会自动计算梯度,无需手工定义
#     return torch.matmul(x.T, y_pred - y)

# 更新参数过程
for i in range(iter_count):
    # forward
    y_pred = forward(x)
    l = loss(y,y_pred)
    print(f'iter {i}, loss {l}')

    # backward
    l.backward()
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()

print(f'final parameter:{w}') # 得到[2,3]

x1 = 4
x2 = 5
# 按公式 y = 2 * x1 + 3 * x2, 则y=23
print(forward(torch.tensor([[x1,x2]],dtype=torch.float32)))
