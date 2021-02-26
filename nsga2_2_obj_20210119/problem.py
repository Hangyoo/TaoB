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
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [450,150,16]  # 决策变量下界
        ub = [5500,2500,24]  # 决策变量上界
        lbin = [0,0,0]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]

        f1 = -1.37320000000054 - 0.000952971428571217*x1 + 0.00323569523809522*x2 + 0.095358333333336*x3 +\
            0.0011151014*x1*x1 + 0.001333953*x1*x2 - 0.0080245613*x1*x3 - 0.0026609584*x2*x2 - 0.00653869047619049*x3*x3

        f2 = 10.7853142857144 - 0.00397828571428574*x1 + 0.00163302857142857*x2 - 0.15904285714286*x3 + 0.00179824994*x1*x1 - 0.0038802749*x1*x2 +\
            0.0257646859*x1*x3 + 0.00163302857142857*x2*x2 - 0.15904285714286*x3*x3

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV