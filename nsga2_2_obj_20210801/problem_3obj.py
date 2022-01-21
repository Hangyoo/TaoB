import numpy as np
import geatpy as ea
import math
import random

"""
目标双目标优化问题
min f1  
min f2
min f3
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 4  # 初始化Dim（决策变量维数）
        maxormins = [-1]*M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0,0,0,0]  # 决策变量下界
        ub = [3,2,2,16]  # 决策变量上界
        lbin = [0]* Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0]* Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数

        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        s2 = Vars[:, [0]]
        s3 = Vars[:, [1]]
        s4 = Vars[:, [2]]
        s5 = Vars[:, [3]]

        u1 = np.array([math.exp(0.1092) for _ in range(len(s2))]).reshape(len(s2),1)
        u2 = np.array([math.exp(0.9117 + 0.608*x) for x in s2]).reshape(len(s2),1)
        u3 = np.array([math.exp(0.2464 + 1.289*x) for x in s3]).reshape(len(s3),1)
        u4 = np.array([math.exp(-0.0926 + 1.797*x) for x in s4]).reshape(len(s4),1)
        u5 = np.array([math.exp(-0.0964 + 0.311*x) for x in s5]).reshape(len(s5),1)
        u = u1 + u2 + u3 + u4 + u5

        N1 = (u1/u) * 1000
        N2 = (u2/u) * 1000
        N3 = (u3/u) * 1000
        N4 = (u4/u) * 1000
        N5 = (u5/u) * 1000


        # 不等式约束条件：
        # 0.1 - (u4/u) <= 0
        CV1 = 0.1 - (u4/u)

        pop.CV = np.hstack([CV1])

        # min
        f1= 8.114*N1 + 6.769*N2 + 6.568*N3 + 6.377*N4 + 6.741*N5
        # min
        f2= s2*N2 + s3*N3 + s4*N4 + s5*N5
        # min
        f3 = 0.0000988*N1 + 0.0000047*N2 + 0.002822*N3 + 0.0017624*N4 + 0.0000872*N5

        pop.ObjV = np.hstack([f1, f2, f3])  # 把求得的目标函数值赋值给种群pop的ObjV
