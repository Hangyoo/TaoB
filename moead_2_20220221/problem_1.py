import numpy as np
import geatpy as ea


"""
最大化单目标的优化问题
max f1 f2 f3 f4 f5
s.t.
x1 - x27 ∈ [0,80000]
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 27  # 初始化Dim（决策变量维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0]*Dim  # 决策变量下界
        ub = [80000]*Dim  # 决策变量上界
        lbin = [1]*Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        x1,x2,x3,x4,x5 = Vars[:, [0]],Vars[:, [1]],Vars[:, [2]],Vars[:, [3]],Vars[:, [4]]
        x6,x7,x8,x9,x10 = Vars[:, [5]],Vars[:, [6]],Vars[:, [7]],Vars[:, [8]],Vars[:, [9]]
        x11,x12,x13,x14,x15 = Vars[:, [10]],Vars[:, [11]],Vars[:, [12]],Vars[:, [13]],Vars[:, [14]]
        x16,x17,x18,x19,x20 = Vars[:, [15]],Vars[:, [16]],Vars[:, [17]],Vars[:, [18]],Vars[:, [19]]
        x21,x22,x23,x24,x25 = Vars[:, [20]],Vars[:, [21]],Vars[:, [22]],Vars[:, [23]],Vars[:, [24]]
        x26 = Vars[:, [25]]
        x27 = Vars[:, [26]]
        # x28 = Vars[:, [27]]

        x3 = 85197 - x1-x2-x4 + 4174
        x10 = 28051 - x8-x9-x11 + 1082
        x17 = 25955 - x15-x16-x18 + 942
        x24 = 32622 - x22-x23-x25 + 498
        x6 = 0.9*48489 + 376 -x5-x7
        x13 = 0.9*36195 + 292 -x12
        x20 = 0.9 * 40022 + 400 -x19
        x27 = 0.9*39382 + 550 - x26

        # 两个目标函数
        f1 = x1+x8+x18+x22
        f2 = (x2+x9+x16+x23) / 10
        f3 = 2*(x3+x10+x17+x24) / 5
        f4 = (x5+x12+x19+x26) / 40
        f5 = (x6+x13+x20+x27) / 10
        f = f1+f2+f3+f4+f5
        pop.ObjV = np.hstack([f])  # 把求得的目标函数值赋值给种群pop的ObjV

        # pop.ObjV = np.hstack([f1,f2,f3,f4,f5])  # 把求得的目标函数值赋值给种群pop的ObjV

        CV1 = 48489 - x4  # <= 0
        CV2 = 40022 - x18
        CV3 = 36195 - x11
        CV4 = 39832 - x25
        pop.CV = np.hstack([CV1, CV2,CV3,CV4])
