import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
max f1 = MRR 
min f2 = Ra

s.t.
0 <= x1 <= 10
150 <= x2 <= 300
1000 <= x3 <= 3000
0 <= x4 <= 150
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [160, 1400, 20]  # 决策变量下界
        ub = [240, 2600, 100]  # 决策变量上界
        lbin = [0, 0, 0]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]

        # Ra
        f1 = 0.412682 - 0.007604 * x1 + 0.000256 * x2 + 0.010968 * x3 - 0.000001 * x1 * x2 - 0.000024 * x1 * x3 - 0.000001 * x2 * x3 + 0.000028 * (
                x1 ** 2) + 0 * (x2 ** 2) - 0.000041 * (x3 ** 2)

        # MRR (f1 = begin*(end*x3+temp1)*(d*x1^2+e*x1+f)*x2)
        a = -1.983495710
        b = 0.000868316
        c = 0.240156715
        d = 0.000000594
        e = -0.000260302
        f = 0.025508826
        f2 = a * (b * x3 + c) * (d * (x1 ** 2) + e * x1 + f) * x2

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV
