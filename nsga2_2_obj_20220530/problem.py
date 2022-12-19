import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 
max f2 

s.t.
140 <= x1 <= 180
210 <= x2 <= 240
50 <= x3 <= 180
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1,-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [140,210,50]  # 决策变量下界
        ub = [180,240,180]  # 决策变量上界
        lbin = [1,1,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]

        # 计算目标函数f1
        f1 = np.pi*(0.508+0.735*1e-3+(x1/3.2-1)*0.805*np.sqrt(3)/2)**2*(2*x3+2*0.735*1e-3+(2*x1-1)*0.805*np.sqrt(3)/2+x2)**2

        f2_1 = 0.04048*x1**2 + 23.61*x1 -1531
        f2_2 = -0.01492*x2**2 + 14.16*x2 -715.9
        f2_3 = -0.0001175*x3**3 + 0.03965*x3**2 - 0.02125*x3 + 2908

        # 计算目标函数f2 取最值
        f2 = np.concatenate([f2_1,f2_2,f2_3],axis=1)
        f2 = f2.max(axis=1).reshape(-1,1)
        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV