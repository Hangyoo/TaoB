import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

data1 = pd.read_csv('loss.csv',header=None)
data2 = pd.read_csv('power.csv',header=None)


size = data1.shape[1]
array = []
for i in range(size):
    a1,a2,a3,a4,a5 = str(data1.iloc[0,i]).split(sep=' ')
    a1_val = float(a1.split("=")[1])
    a2_val = float(a2.split("=")[1])
    a3_val = float(a3.split("=")[1])
    a4_val = float(a4.split("=")[1])
    a5_val = float(a5.split("=")[1])
    loss = data1.iloc[1,i]
    power = data2.iloc[1,i]
    array.append([a1_val,a2_val,a3_val,a4_val,a5_val,loss,power])

array = np.array(array).reshape((size,7))
# 经过预处理后的数据
data = pd.DataFrame(array)
data.to_excel('data.xls',index=None)
print('数据预处理完毕！')
