import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self):
        txt_data = np.loadtxt(r'C:\Users\Hangyu\PycharmProjects\TaoB\ML1\sample_data',delimiter=',')
        self._x = torch.from_numpy(txt_data[:,:2])
        self._y = torch.from_numpy(txt_data[:,2])
        self._len = len(txt_data)

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len

data = MyDataset()
print(len(data))

first = next(iter(data))
print(first)

dataloader = DataLoader(data,batch_size=3,shuffle=True,drop_last=True) # 不能整数的话把最后的数据drop掉
n = 0
for data_val, label_val in dataloader:
    print("x:",data_val,"y:",label_val)
    n += 1
print("iteration:",n)

