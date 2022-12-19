import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''函数功能: 绘制迭代前算法种群目标值 和 迭代后算法PF'''


# 读取迭代前目标
data_before = pd.read_csv('Obj_before.csv',header=None)
x_before = data_before.iloc[:,0].tolist()
y_before = data_before.iloc[:,1].tolist()

# 读取迭代后目标
data_after = pd.read_csv('Objective.csv',header=None)
x_after = data_after.iloc[:,0].tolist()
y_after = data_after.iloc[:,1].tolist()

plt.scatter(x_before,y_before,color='blue')
plt.scatter(x_after,y_after,color='red')
plt.legend(["Before Optimization","After Optimization"])
plt.show()