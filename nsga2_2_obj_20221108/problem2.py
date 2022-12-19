import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 环境影响
max f2 光伏发电量

s.t.
0 <= x1 <= 1105000000
0 <= x2 <= 274000000
F2 < 14*10^10
"""

E1 = 5.173108293  # 地面土地占用
E2 = 1.382880143  # 屋顶土地占用

EF = 0.15
PR = 0.7
GHI = 1420

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 2  # 初始化Dim（决策变量维数）
        maxormins = [1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0,0]  # 决策变量下界
        ub = [1105000000,274000000]  # 决策变量上界
        lbin = [1,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        x1 = Vars[:, [0]]  # A1
        x2 = Vars[:, [1]]  # A2

        f1 = x1 * E1 + x2 * E2
        f2 = (x1 + x2) * GHI *PR * EF

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV

        # 约束处理 F2 <= 14*E10
        CV1 = f2 - 14e10
        pop.CV = np.hstack([CV1])