import pandas as pd
import warnings
import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 = 0.283Ia + 0.154Ib + 0.158Ic - 15264.09
min f2 = 0.246Ia + 0.0068Ib + 0.122Ic - 15263.9
min f3 = 0.177Ia + 0.154Ib + 0.1Ic - 31.38
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [-500, 0, -500]  # 决策变量下界
        ub = [0, 1000, 0]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        Ia = Vars[:, [0]]
        Ib = Vars[:, [1]]
        Ic = Vars[:, [2]]

        f1 = 0.283 * Ia + 0.154 * Ib + 0.158 * Ic - 15264.09

        f2 = 0.246 * Ia + 0.0068 * Ib + 0.122 * Ic - 15263.9

        f3 = 0.177 * Ia + 0.154 * Ib + 0.1 * Ic - 31.38

        pop.ObjV = np.hstack([f1, f2, f3])  # 把求得的目标函数值赋值给种群pop的ObjV
