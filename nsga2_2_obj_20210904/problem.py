import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1
min f2

s.t.
0.3 <= f <= 3.0
0 <= L <= 15
10 <= ap <= 25
0.22 <= vc <= 3.85
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 4  # 初始化Dim（决策变量维数）
        maxormins = [1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # todo varTypes 中的1可以尝试换成0
        varTypes = [1] * Dim

        lb = [0.3, 0.1, 10, 0.22]  # 决策变量下界
        ub = [3.0, 15, 25, 3.85]  # 决策变量上界
        lbin = [1, 1, 1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        f = Vars[:, [0]] * 0.001   # mm 转换成 m
        L = Vars[:, [1]]
        ap = Vars[:, [2]] * 0.001   # mm 转换成 m
        vc = Vars[:, [3]]

        f1 = 3 * ap * vc * (f ** 0.75)

        f2 = 0.06 * L / (vc * f * ap) + 1440 + f * vc / 2060

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV
