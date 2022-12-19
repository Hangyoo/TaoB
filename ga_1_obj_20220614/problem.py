import numpy as np
import geatpy as ea
import math

"""


最小化目标单目标优化问题 
3.0<=vs<=3.3
6.35<=Dp<=7.05
2800<=Dcap<=3200
"""

ro = 1.0259
J = 0.33
KQ = 0.278

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 3  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [3.0, 6.35, 2800]  # 决策变量下界
        ub = [3.3, 7.05, 3200]  # 决策变量上界
        lbin = [1, 1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        vs = Vars[:, [0]]
        Dp = Vars[:, [1]]
        Dcap = Vars[:, [2]]

        MCR = (2*np.pi*ro*vs*vs*0.74*0.74*KQ*Dp**3/J) / (0.985*0.985)
        PME = 0.75 * MCR
        EEDI_attend = 3.114*(163.61 * PME + 193*521.25) / (Dcap*17.39)
        # EEDI_lvr = 3.114*((190 * PME) + 215*521.25) / (Dcap*17.39)
        EEDI_lvr = 9.898

        f1 = EEDI_attend / EEDI_lvr - 1

        pop.ObjV = np.hstack([f1])  # 把求得的目标函数值赋值给种群pop的ObjV

