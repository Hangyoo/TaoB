import pandas as pd
import warnings
import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 
min f2  
"""

A = 168
B = 10.263
v0 = 19
Phfo = 405
Pmfo = 609
D = 23142

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 1  # 初始化Dim（决策变量维数）
        maxormins = [1,1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [8.46]   # 决策变量下界
        ub = [22.5]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)

        v = Vars[:, [0]]


        f1 = (A*(v/v0)**3)*(D/(24*v))*Phfo + B*(D/(24*v))*Pmfo

        f2 = D / (24 * v)

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV


