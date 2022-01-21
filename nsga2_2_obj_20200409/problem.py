import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 = WIC (自己定义的函数)
min f2 = Nij

s.t.
4500<= x1 <= 5500
1500<= x2 <= 2500
16ms<= x3 <= 24
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 2  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [760,4.8]  # 决策变量下界
        ub = [2280,11.3]  # 决策变量上界
        lbin = [1,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]

        f1 = (2.15e-8) * x1 * x1 * x1 + (1.65e-6) * x1 * x1 * x2 - (2.1e-6) * x1 * x2 * x2 + 1.05 * x2 * x2 * x2 - (
            6.9e-5) * x1 * x1 - (5.83e-3) * x1 * x2 - \
             23.96 * x2 * x2 + 0.07 * x1 + 176.86 * x2 + 89.59

        f2 = (1.75e-8) * x1 * x1 * x1 + (2.95e-6) * x1 * x1 * x2 + (1.1e-3) * x1 * x2 * x2 + 1.51 * x2 * x2 * x2 - (
            9.1e-5) * x1 * x1 - 0.03 * x1 * x2 - \
             36.34 * x2 * x2 + 0.22 * x1 + 280.59 * x2 - 121.31

        f3 = -(6.7e-7) * x1 * x1 - (4.5e-4) * x1 * x2 + 0.29 * x2 * x2 - (2.43e-2) * x1 + 1.84 * x2 + 518.91

        pop.ObjV = np.hstack([f1, f2, f3])  # 把求得的目标函数值赋值给种群pop的ObjV