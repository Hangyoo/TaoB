import numpy as np
import geatpy as ea

"""
决策变量11个 x1 - x11

最小化目标双目标优化问题
min f1 = x1+x2+x3-x4-x5-x6-x7-x8-x9-x10-x11
max f2 = alpha*x9 + beta
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 11  # 初始化Dim（决策变量维数）
        maxormins = [1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）

        # x1up = input('请输入产生单元x1的上限：')
        # x2up = input('请输入产生单元x2的上限：')
        # x3up = input('请输入产生单元x3的上限：')
        # x4up = input('请输入消耗单元x4的上限：')
        # x5up = input('请输入消耗单元x5的上限：')
        # x6up = input('请输入调节单元x6的上限：')
        # x7up = input('请输入调节单元x7的上限：')
        # x8up = input('请输入调节单元x8的上限：')
        x1up = 250000
        x2up = 250000
        x3up = 250000
        x4up = 30000
        x5up = 30000
        x6up = 30000
        x7up = 25000
        x8up = 30000
        lb = [0]*Dim  # 决策变量下界
        ub = [x1up, x2up, x3up, x4up, x5up, x6up, x7up, x8up, 80000, 80000, 15000]# 决策变量上界

        lbin = [1]*Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
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
        x11 = Vars[:, [10]]

        x12 = x1+x2+x3-x4-x5-x6-x7-x8-x9-x10-x11
        # 两个目标函数
        f1 = x12
        f2 = 0.8*x8 + 1000
        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV

        # 约束处理 (如果有约束可以写在这里)

        CV1 = -x12  # x12 >=0 的约束
        pop.CV = np.hstack([CV1])
