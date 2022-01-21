import numpy as np
import geatpy as ea

"""
最小化目标单目标优化问题
"""

# 数据
T_ni = [25, 23, 15, 25, 10, 23, 12, 12, 12, 12, 12, 12, 20, 15, 10, 10, 10, 10, 10, 30, 35, 35, 35, 25, 30, 7]
C_ni = [6.81, 22.06, 10.57, 89.2, 6.95, 50.32, 48.17, 47.99, 49.99, 47.99, 44.76, 44.29, 7.31, 34.06, 31.73,
        31.63, 31.63, 31.66, 31.66, 46.55, 129.89, 136.2, 138.06, 158.64, 265.78, 3.79]
T_si = [20, 18, 12, 18, 10, 18, 10, 10, 10, 10, 10, 10, 14, 12, 8, 8, 8, 8, 8, 26, 32, 28, 30, 22, 26, 7]
C_si = [8.12, 29.16, 13.22, 96.97, 6.95, 57.17, 51.95, 50.77, 50.77, 50.77, 47.54, 47.07, 14.15, 37.48, 34.01,
        33.91, 33.91, 33.94, 33.94, 51.83, 133.13, 142.98, 143.96, 162.21, 269.47, 3.79]
gamma = 0.3  # 提前完工奖励
beta = 0.39  # 项目间接费用率


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 26  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [20, 18, 12, 18, 10, 18, 10, 10, 10, 10, 10, 10, 14, 12, 8, 8, 8, 8, 8, 26, 32, 28, 30, 22, 26,7]  # 决策变量下界
        ub = [25, 23, 15, 25, 10, 23, 12, 12, 12, 12, 12, 12, 20, 15, 10, 10, 10, 10, 10, 30, 35, 35, 35, 25, 30,7]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")] * popsize).reshape(popsize, 1)

        alpha = []
        for k in range(26):
            alpha.append((C_si[k] - C_ni[k]) / ((T_ni[k] - T_si[k]) ** 2 + 0.01))  # todo 这里改alpha

        Ci = []
        for i in range(popsize):
            for j in range(26):
                ti = Vars[i, [j]][0]
                ci = C_ni[j] + alpha[j] * ((T_ni[j] - ti) ** 2) + beta * ti
                Ci.append(ci)
            F1[i, 0] = sum(Ci)

        pop.ObjV = np.hstack([F1])  # 把求得的目标函数值赋值给种群pop的ObjV
