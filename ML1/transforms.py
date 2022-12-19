import torch
from torchvision import transforms

class ScaleTransform():

    def __init__(self,scale_factor):
        self._scale_factor = scale_factor

    def __call__(self, sample):
        sample = sample * self._scale_factor
        return sample

data = torch.rand(3,5)
scale_transform = ScaleTransform(10)  # 重写了__call__函数,所以instance可以当做一个function来用
data_after_scale = scale_transform(data)

print(f'origin data:\n{data}')
print(f'data after scale:\n{data_after_scale}')


class AddTransform():

    def __init__(self,add_value):
        self._add_value = add_value

    def __call__(self, sample):
        sample = sample + self._add_value
        return sample

add_transform = AddTransform(50) # 让每个值都加上50
data_after_add = add_transform(data_after_scale)
print(f'data after scale:\n{data_after_add}')


# 多个transform整合成为一个transform
compose_transform = transforms.Compose([scale_transform,add_transform])
data_after_compose_transform = compose_transform(data)
print(f"data after composed transform:\n{data_after_compose_transform}")


