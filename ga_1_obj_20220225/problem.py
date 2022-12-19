import numpy as np
import geatpy as ea
import math

"""
最小化目标单目标优化问题 
60<=x<=100
60<=y<=100
30<=z<=50
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [60, 60, 30]  # 决策变量下界
        ub = [100, 100, 50]  # 决策变量上界
        lbin = [1, 1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        x = Vars[:, [0]]
        y = Vars[:, [1]]
        z = Vars[:, [2]]

        f1 = 10 * 7.85 * (10e-6) * (math.pi / 4) * (362 * x * 2 + 378 * y ** 2 + 44 * z ** 2) -185

        pop.ObjV = np.hstack([f1])  # 把求得的目标函数值赋值给种群pop的ObjV

        # 约束1
        CV1 = -381.09728 + 2.75068 * x - 0.1041 * y - 1.10263 * z - 320  # <= 0
        # 约束2
        CV2 = 100 - 858.4 + 8.036 * x + 0.424 * y + 3.52 * z
        CV3 = 858.4 - 8.036 * x - 0.424 * y - 3.52 * z - 668
        # 约束3
        CV4 = 100 - 188.00533 - 1.38867 * x + 2.7194 * y + 0.3613 * z
        CV5 = 188.00533 + 1.38867 * x - 2.7194 * y - 0.3613 * z - 668
        pop.CV = np.hstack([CV1, CV2, CV3, CV4, CV5])
