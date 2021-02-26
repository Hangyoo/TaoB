import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
max f1 = MRR 
min f2 = Ra

s.t.
0 <= x1 <= 100 (As)
0 <= x2 <= 100 (Ab)
0 <= x3 <= 100 (Az)
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0, 0, 0]  # 决策变量下界
        ub = [100, 100, 100]  # 决策变量上界
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
        # 两个目标函数
        f1 = 100 / ((25 * x1 + 5.6 * x2 + 4.6 * x3) * 0.115)
        f2 = 1/(180 * x1 + 45 * x2 + 78 * x3)
        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV

        # 约束处理
        Ag = (3 + ((5.38 * x1 + 3 * x2 + 1.82 * x3) * 0.115) / (12 * 38 * 100)) * 15 * 100
        CV1 = 0.5 - Ag / (x1 + x2 + x3)
        CV2 = 0.55 - x1 / (x1 + x2 + x3)
        CV3 = (Ag + x1 + x2 + x3) / 4163.5 - 3.5
        CV4 = ((5.38 * x1 + 3 * x2 + 1.82 * x3) * (2 * 0.115 + 0.166)) / 100 - 242
        pop.CV = np.hstack([CV1, CV2, CV3, CV4])
