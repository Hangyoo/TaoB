import numpy as np
import pickle
import geatpy as ea
import math
from RF_NSGA2_20220417.surrogate import ss_x, ss_y1, ss_y2

"""
决策变量的最小化目标双目标优化问题
min f1
min f2
min f3

"""

# 读取训练好的机器学习模型
with open("model1.pkl", "rb") as f1:  # 预测loss
    model1 = pickle.load(f1)

with open("model2.pkl", "rb") as f2:  # 预测 power
    model2 = pickle.load(f2)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 9  # 初始化Dim（决策变量维数）
        maxormins = [1, 1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [3, 65, 23, 10, 2.5, 9, 0, 10, 58]  # 决策变量下界
        ub = [18, 90, 65, 35, 8, 15, 3, 26, 80]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F2 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F3 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            T1 = Vars[i, [0]][0]
            # theta1 = Vars[i, [1]][0]
            T2 = Vars[i, [2]][0]
            theta2 = Vars[i, [3]][0]
            Dt = Vars[i, [4]][0]
            Ht = Vars[i, [5]][0]
            Db = Vars[i, [6]][0]
            Dj = Vars[i, [7]][0]
            Hj = Vars[i, [8]][0]
            # theta1 + theta2 = 90
            theta1 = 90 - theta2
            Vars[i, [1]][0] = theta1
            # 对x进行归一化操作，使其在0-1之间
            X = ss_x.transform(np.array([T1, theta1, T2, theta2, Dt, Ht, Db, Dj, Hj]).reshape(1, 9))
            # 计算目标函数值
            f1 = model1.predict(X)  # loss
            f2 = model2.predict(X)  # power
            f3 = (math.pi / 4) * (Dt ** 2) * Ht + (math.pi / 4) * (Dj ** 2) * Hj
            F1[i, 0] = ss_y1.inverse_transform(f1)
            F2[i, 0] = ss_y2.inverse_transform(f2)
            F3[i, 0] = f3
        pop.ObjV = np.hstack([F1, F2, F3])  # 把求得的目标函数值赋值给种群pop的ObjV

        # 添加约束 theta1/T1 > theta2/T2  即 theta2/T2 - theta1/T1 < 0
        theta1 = Vars[:, [1]]
        T1 = Vars[:, [0]]
        theta2 = Vars[:, [3]]
        T2 = Vars[:, [2]]

        CV1 = theta2/T2 - theta1/T1  # <= 0
        pop.CV = np.hstack([CV1])
