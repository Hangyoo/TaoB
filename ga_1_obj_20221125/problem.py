import numpy as np
import geatpy as ea
import math

"""
最小化目标单目标优化问题 
3.0<=k<=6.0
0.4<=Q<=0.6
0.6<=fn<=0.75
"""

n = 1
Uin = 200
R0 = 48.4

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [3.0, 0.4, 0.6]  # 决策变量下界
        ub = [6.0, 0.6, 0.75]  # 决策变量上界
        lbin = [1, 1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        K = Vars[:, [0]]
        Q = Vars[:, [1]]
        fn = Vars[:, [2]]

        Req = 8 * n ** 2 * R0 / math.pi ** 2
        A = (math.pi * Uin / (2 * np.sqrt((1 + 1 / K * (1 - 1 / fn ** 2)) ** 2 + Q ** 2 * (fn - 1 / fn) ** 2))) / (
                    2 * K * Q * Req)
        B = math.pi / np.sqrt(1 + K) * (1 / fn - 1)
        C = (math.pi * fn * Uin / 2 - 4 * Uin * Q / (
                    2 * np.sqrt((1 + 1 / K * (1 - 1 / fn ** 2)) ** 2 + Q ** 2 * (fn - 1 / fn) ** 2))) / (
                        math.pi * fn * Q * Req * np.sqrt(1 + K))
        D = B
        Ioff = A * np.cos(B) + C * np.sin(D)


        pop.ObjV = np.hstack([Ioff])  # 把求得的目标函数值赋值给种群pop的ObjV

