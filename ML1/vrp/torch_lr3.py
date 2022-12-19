import torch
import numpy as np

'''
在torch_linear_regression的基础上改写
'''

'''用lOSS'''
x = torch.tensor([[1,2],[2,3],[4,6],[3,1]],dtype=torch.float32)
y = torch.tensor([[8],[13],[26],[9]],dtype=torch.float32)
# y = 2 * x1 + 3 * x2

w = torch.rand(2,1,requires_grad=True,dtype=torch.float32)
iter_count = 300
lr = 0.01

# def forward(x):
#     return torch.matmul(x,w) # (4.2) * (2*1) --> (4,1)

# def loss(y,y_pred): pytorch中有MSEloss
#     return ((y-y_pred)**2 / 2).sum()

# def gradient(x,y,y_pred): torch 会自动计算梯度,无需手工定义
#     return torch.matmul(x.T, y_pred - y)

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = torch.nn.Parameter(torch.rand(2,1,dtype=torch.float32))

    def forward(self,x):
        return torch.matmul(x,self.w) # (4.2) * (2*1) --> (4,1)

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(),lr)

# 更新参数过程
criterion = torch.nn.MSELoss(reduction='sum') # pytorch定义损失函数

for i in range(iter_count):
    # forward
    y_pred = model(x)
    # l = loss(y,y_pred)
    l = criterion(y_pred,y)
    print(f'iter {i}, loss {l}')

    # backward
    l.backward()
    optimizer.step() # 更新参数
    optimizer.zero_grad() # 梯度置零
    # with torch.no_grad():  # 让optimizer来替换这个部分
    #     w -= lr * w.grad
    #     w.grad.zero_()

print(f'final parameter:{model.w}') # 得到[2,3]

x1 = 4
x2 = 5
# 按公式 y = 2 * x1 + 3 * x2, 则y=23
print(model(torch.tensor([[x1,x2]],dtype=torch.float32)))