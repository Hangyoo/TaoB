import numpy as np
import geatpy as ea

from nsgaii_2_obj_20201202.pollute import funtion2


"""
离散决策变量的最小化目标双目标优化问题
min f1 = 自己定义的函数
min f2 = 50*x1 + 50*x2 + 50*x3

s.t.
x1 ∈ {1,2,3,4,5}
x2 ∈ {1,2,3,4}
x3 ∈ {1,2,3,4,5,6}
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [1] * Dim  # 决策变量下界
        ub = [5,4,6]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        print(len(x1))
        f1 = funtion2(x1,x2,x3)
        f2 = 50*x1 + 50*x2 + 50*x3


        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV