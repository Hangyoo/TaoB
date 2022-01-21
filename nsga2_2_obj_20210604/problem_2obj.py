import numpy as np
import geatpy as ea
import random

"""
目标双目标优化问题
min f1  
min f2 
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 10  # 初始化Dim（决策变量维数）
        maxormins = [-1]*M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [128529,1480000,26000,56900,-5000,340,54700,1050,18370,23700]  # 决策变量下界
        ub = [172168.83,1490000,26800,60000,8370,500,69000,1750,19130,30800]  # 决策变量上界
        lbin = [0]* Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0]* Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数

        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        x5 = Vars[:, [4]]
        x6 = Vars[:, [5]]
        x7 = Vars[:, [6]]
        x8 = Vars[:, [7]]
        x9 = Vars[:, [8]]
        x10 = Vars[:, [9]]

        # 不等式约束条件：
        # 1500000 - (x(2) + x(3) + x(4) + x(5) + x(6)) < 0
        CV1 = 1500000 - (x2+x3+x4+x5+x6)

        # 等式约束条件：
        # x(1) + x(2) + x(3) + x(4) + x(5) + x(6) + x(7) + x(8) + x(9) + x(10) = 1876897
        # 转换成 x(1) + x(2) + x(3) + x(4) + x(5) + x(6) + x(7) + x(8) + x(9) + x(10) - 1876897 >= 0
        # 和    x(1) + x(2) + x(3) + x(4) + x(5) + x(6) + x(7) + x(8) + x(9) + x(10) - 1876897 <= 0
        CV2 = 1876897 - (x1+x2+x3+x4+x5+x6+x7+x8+x9+x10)
        CV3 = (x1+x2+x3+x4+x5+x6+x7+x8+x9+x10) - 1876897
        pop.CV = np.hstack([CV1, CV2, CV3])

        # min
        f1=-1.93*x1-0.03*x2-0.03*x3-0.03*x4-0.03*x5-0.03*x6-1.71*x7-0.59*x8-0.59*x9-99.57*x10
        # min
        f2=-1.17*x1-4.04*x2-3.09*x3-4.06*x4-3.37*x5-2.68*x6-2.12*x7-11.96*x8-11.96*x9-1.16*x10
        # min
        # f3 = -0.244*x1 - 0.332*x2 - 0.316*x3 - 0.355*x4 - 0.258*x5 - 0.321*x6 - 0.275*x7 - 0.3*x8 + 0.12*x9 + 0*x10

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV
