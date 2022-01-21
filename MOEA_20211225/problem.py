import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1
min f2
min f3

s.t.
0.75 <= x1 <= 2.95  风室1的取值范围
0.75 <= x2 <= 2.95  风室2的取值范围
0.75 <= x3 <= 2.95  风室3的取值范围
0.75 <= x4 <= 2.95  风室4的取值范围
0.75 <= x5 <= 2.95  风室5的取值范围
0.75 <= x6 <= 2.95  风室6的取值范围
"""

Temp = []
Temp_Var = []


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 6  # 初始化Dim（决策变量维数）
        maxormins = [-1, 1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        # lb = [0.75] * Dim  # 决策变量下界
        # ub = [2.95] * Dim  # 决策变量上界
        lb = [0.95]*Dim   # 决策变量下界
        ub = [2.95]*Dim   # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F2 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F3 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            x1 = Vars[i, [0]][0]
            x2 = Vars[i, [1]][0]
            x3 = Vars[i, [2]][0]
            x4 = Vars[i, [3]][0]
            x5 = Vars[i, [4]][0]
            x6 = Vars[i, [5]][0]

            # 二次风温
            f1 = 1065.48 - 470.959 * x1 + 769.657 * x2 - 232.594 * x3 + 45.2741 * x4 - 146.575 * x5 + 19.2488 * x6 + \
                 165.029 * x1 * x2 + 114.252 * x1 * x3 + 216.278 * x1 * x4 - 314.353 * x1 * x5 + 168.586 * x1 * x6 - \
                 370.033 * x2 * x3 - 526.56 * x2 * x4 + 642.312 * x2 * x5 - 180.901 * x2 * x6 + 414.795 * x3 * x4 - \
                 517.495 * x3 * x5 + 66.9567 * x3 * x6 + 669.121 * x4 * x5 - 302.959 * x4 * x6 + 277.075 * x5 * x6 - \
                 48.2285 * x1 ** 2 - 80.8058 * x2 ** 2 + 197.09 * x3 ** 2 - 253.144 * x4 ** 2 - 315.906 * x5 ** 2 - 25.1113 * x6 ** 2

            # 出温风温
            f2 = 464.612 - 400.363 * x1 + 674.382 * x2 - 318.308 * x3 - 137.079 * x4 + 163.213 * x5 - 144.573 * x6 + \
                 74.0726 * x1 * x2 + 202.806 * x1 * x3 - 18.3681 * x1 * x4 - 28.6151 * x1 * x5 + 3.61567 * x1 * x6 - \
                 426.006 * x2 * x3 + 56.2006 * x2 * x4 + 37.6291 * x2 * x5 + 72.1038 * x2 * x6 - 23.1415 * x3 * x4 + \
                 48.7089 * x3 * x5 - 152.869 * x3 * x6 - 196.325 * x4 * x5 + 126.504 * x4 * x6 + 89.7283 * x5 * x6 + \
                 3.30492 * x1 ** 2 - 112.089 * x2 ** 2 + 259.391 * x3 ** 2 + 66.221 * x4 ** 2 - 9.25738 * x5 ** 2 - 35.7172 * x6 ** 2

            # 出口温度
            f3 = 332.285 - 3.76115 * x1 - 57.2402 * x2 - 7.03187 * x3 - 185.967 * x4 + 164.27 * x5 - 138.834 * x6 - \
                 31.0463 * x1 * x2 + 63.3426 * x1 * x3 - 77.3782 * x1 * x4 - 12.2009 * x1 * x5 + 50.492 * x1 * x6 - \
                 98.1579 * x2 * x3 + 117.233 * x2 * x4 + 45.8926 * x2 * x5 - 86.4151 * x2 * x6 + 9.31394 * x3 * x4 - \
                 26.8711 * x3 * x5 - 12.3141 * x3 * x6 - 305.437 * x4 * x5 + 228.191 * x4 * x6 - 50.4836 * x5 * x6 + \
                 9.53542 * x1 ** 2 + 32.4082 * x2 ** 2 + 32.4597 * x3 ** 2 + 67.2034 * x4 ** 2 + 132.538 * x5 ** 2 - 30.5995 * x6 ** 2

            penalty = 500
            if f1 < 1000 or f1 > 1200 or f2 < 250 or f2 > 320 or f3 < 90 or f3 > 150:
                f1 -= penalty
                f2 += penalty
                f3 += penalty

            if 1000 <= f1 <= 1200 and 250 <= f2 <= 320 and 30 <= f3 <= 150:
                if [f1,f2,f3] not in Temp:
                    Temp.append([f1,f2,f3])
                    Temp_Var.append([pop.Phen[i, [0]][0],pop.Phen[i, [1]][0],pop.Phen[i, [2]][0],
                                     pop.Phen[i, [3]][0],pop.Phen[i, [4]][0],pop.Phen[i, [5]][0]])

            F1[i, 0] = f1
            F2[i, 0] = f2
            F3[i, 0] = f3

        pop.ObjV = np.hstack([F1, F2, F3])  # 把求得的目标函数值赋值给种群pop的ObjV


    def save(self):
        objset = []
        varset = []
        for i in range(len(Temp)):
            item = Temp[i]
            if item not in objset:
                objset.append(item)
                varset.append(Temp_Var[i])
        return objset,varset
