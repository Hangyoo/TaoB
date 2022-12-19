import torch

'''用pytorch来计算gradient'''
# calculate: z = 2 * a + a * b

a = torch.tensor(3., requires_grad=True) # 这里不能是整数, 如果是整数会报错
b = torch.tensor(4.)

print(a.requires_grad) # 查看是否需要对a求导数
print(b.requires_grad) # 查看是否需要对b求导数

b.requires_grad_(True)
print(b.requires_grad)

f1 = 2*a
f2 = a*b
z = f1 + f2
print(z)

z.backward() # 计算导数
print(f"a.grad:{a.grad}") # z对a的导数
print(f"b.grad:{b.grad}") # z对b的导数


# 无需计算导数的方法 (2种)
with torch.no_grad():
    f3 = a*b
    print(f"f2.requires_grad={f2.requires_grad}")  # 会对f2计算梯度
    print(f"f3.requires_grad={f3.requires_grad}")  # 不会对f3计算梯度

a1 = a.detach() # 从计算图中分离出来
print(f"a.requires_grad={a.requires_grad}")
print(f"a1.requires_grad={a1.requires_grad}")