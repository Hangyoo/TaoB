import random

import numpy as np
import pickle
import geatpy as ea
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings

# 读取数据
warnings.filterwarnings("ignore")
data = pd.read_excel(r"C:\Users\Hangyu\Desktop\0311\data.xls")
data_inputs = np.array(data.iloc[:,0:3])
# 读取数据标签(100*2)
f1 = np.array(data.iloc[:,3:4])
f2 = np.array(data.iloc[:,4:5])
f3 = np.array(data.iloc[:,4:5])
f4 = np.array(data.iloc[:,5:6])
ss_x = StandardScaler() # 实例化用于对特征标准化类
ss_y1 = StandardScaler() # 实例化用于对标签标准化类
ss_y2 = StandardScaler() # 实例化用于对标签标准化类
ss_y3 = StandardScaler() # 实例化用于对标签标准化类
ss_y4 = StandardScaler() # 实例化用于对标签标准化类
# 对数据进行标准化
data_inputs = ss_x.fit_transform(pd.DataFrame(data_inputs))
# 读取数据标签
f1 = ss_y1.fit_transform(pd.DataFrame(f1))
f2 = ss_y2.fit_transform(pd.DataFrame(f2))
f3 = ss_y3.fit_transform(pd.DataFrame(f3))
f4 = ss_y4.fit_transform(pd.DataFrame(f4))

# 读取训练好的rbf模型
with open("model1.pkl","rb") as f1:
    model1 = pickle.load(f1)

with open("model2.pkl", "rb") as f2:
    model2 = pickle.load(f2)

with open("model3.pkl", "rb") as f3:
    model3 = pickle.load(f3)

with open("model4.pkl", "rb") as f4:
    model4 = pickle.load(f4)

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=4):
        name = 'MyProblem'    # 初始化name（函数名称，可以随意设置）
        Dim = 3               # 初始化Dim（决策变量维数）
        maxormins = [1,-1,1,-1]   # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1,1,0]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [1000,5,0.3]       # 决策变量下界
        ub = [1600,10,1.5]     # 决策变量上界
        lbin = [1] * Dim      # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim      # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)

        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        F2 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        F3 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        F4 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        for i in range(popsize):
            x1 = Vars[i, [0]][0]
            x2 = Vars[i, [1]][0]
            x3 = Vars[i, [2]][0]

            X = ss_x.transform(np.array([x1,x2,x3]).reshape(1,3))

            # 计算目标函数值
            f1 = model1.predict(X)
            f2 = model2.predict(X)
            f3 = model3.predict(X)
            f4 = model4.predict(X)
            f1 = ss_y1.inverse_transform(f1)

            f2 = ss_y2.inverse_transform(f2)
            f3 = ss_y3.inverse_transform(f3)
            f4 = ss_y4.inverse_transform(f4)



            F1[i,0] = f1
            F2[i,0] = f2
            F3[i,0] = f3
            F4[i,0] = f4
        pop.ObjV = np.hstack([F1, F2,F3, F4])  # 把求得的目标函数值赋值给种群pop的ObjV
