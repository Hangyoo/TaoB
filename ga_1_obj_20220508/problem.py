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
        Dim = 7  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0, 0, 0,0,0,0,0]  # 决策变量下界
        ub = [5000, 5000, 5000,5000, 5000, 5000,5000]  # 决策变量上界
        lbin = [1, 1, 1,1,1,1,1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1,1,1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        x5 = Vars[:, [4]]
        x6 = Vars[:, [5]]
        x7 = Vars[:, [6]]

        f1 = x1+x2+x3+x4+x5+x6+x7

        pop.ObjV = np.hstack([f1])  # 把求得的目标函数值赋值给种群pop的ObjV

        # 约束1
        CV1 = 0.11 - (0.134*x1+0.145*x2+0.148*x3+0.152*x4+0.133*x6+0.142*x7) / (x1+x2+x3+x4+x5+x6+x7)
        CV2 =  (0.134*x1+0.145*x2+0.148*x3+0.152*x4+0.133*x6+0.142*x7) / (x1+x2+x3+x4+x5+x6+x7) - 0.13
        # 约束2
        CV3 = 5500 - (x1+x2+x3+x4+x5+x6+x7)
        CV4 = x1+x2+x3+x4+x5+x6+x7 - 7300
        # 约束3
        CV5 = 0.3 - (x1+x3+x5)/(x1+x2+x3+x4+x5+x6+x7)
        CV6 = (x1+x3+x5)/(x1+x2+x3+x4+x5+x6+x7) - 0.5
        # 约束4
        CV7 = 0.2 - (x2+x4)/(x1+x2+x3+x4+x5+x6+x7)
        CV8 = (x2+x4)/(x1+x2+x3+x4+x5+x6+x7) - 0.3

        pop.CV = np.hstack([CV1, CV2, CV3, CV4, CV5, CV6, CV7, CV8])
