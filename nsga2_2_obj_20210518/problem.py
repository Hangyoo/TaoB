import numpy as np
import geatpy as ea
import random

"""
目标双目标优化问题
max f1  
min f2 
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 1  # 初始化Dim（决策变量维数）
        maxormins = [-1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0]  # 决策变量下界
        ub = [1]  # 决策变量上界
        lbin = [1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数

        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        x = Vars[:, [0]]
        q = [20, 25, 18, 15, 23]
        Q = [20, 22, 27, 15, 23]
        Bi = [0.99, 0.9, 0.95, 0.71, 0.75]
        C = 10
        Po = 0.5
        Pd = 0.75
        P = [Po + (bi - 0.8) * Po * x for bi in Bi]

        for i in range(len(P)):
            if (P[i][0] < 0) or (P[i][0] > Pd):
                val = random.random()-0.25 if random.random()-0.25 else 0.5
                P[i][0] = val


        # max
        f1 = q[0] * P[1] + q[1] * P[2] + q[3] * P[0] + q[3] * P[3] + q[4] * P[4] - C
        # min
        f2 = q[0]*P[1]+q[1]*P[2]+q[2]*P[0]+q[3]*P[3]+q[4]*P[4]+(q[3]-Q[3]*Bi[3])+(q[4]-Q[4]*Bi[4])*Pd

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV
