import numpy as np
import pickle
import geatpy as ea
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ga_svm_20220329.surrogate import ss_x,ss_y


"""
最大化单目标的优化问题
min f1
"""

# 获得每个特征的最大值和最小值
UB = list(pd.read_excel("data.xls").max())
LB = list(pd.read_excel("data.xls").min())

print(LB)
print(UB)

# 读取训练好的SVR机器学习模型
with open("model.pkl","rb") as f1:
    model1 = pickle.load(f1)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'    # 初始化name（函数名称，可以随意设置）
        Dim = 9               # 初始化Dim（决策变量维数）
        maxormins = [-1] * M   # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = LB[1:10]        # 决策变量下界
        ub = UB[1:10]         # 决策变量上界
        lbin = [1] * Dim      # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim      # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")]*popsize).reshape(popsize,1)

        for i in range(popsize):
            # 获取特征
            x1 = Vars[i, 0]
            x2 = Vars[i, 1]
            x3 = Vars[i, 2]
            x4 = Vars[i, 3]
            x5 = Vars[i, 4]
            x6 = Vars[i, 5]
            x7 = Vars[i, 6]
            x8 = Vars[i, 7]
            x9 = Vars[i, 8]
            # 对x进行归一化操作，使其在0-1之间
            X = ss_x.transform(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9]).reshape(1,9))
            # 计算目标函数值
            f1 = model1.predict(X)
            # 预测值反归一化
            f1 = ss_y.inverse_transform(f1)
            # 记录目标值
            F1[i,0] = f1
        pop.ObjV = F1  # 把求得的目标函数值赋值给种群pop的ObjV
