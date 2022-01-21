import math
import warnings
import numpy as np
import geatpy as ea

"""
min f1 
min f2 
min f3 
min f4 
min f5 

s.t.
0 < L1 <= 0.00497
0 < L2 <= 0.00249
0 < C <= 0.00003288
0.3 <= R <= 0.8
"""

# 定义常量
warnings.filterwarnings('ignore')

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=5):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 4  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [1e-10,1e-10,1e-10,0.3]  # 决策变量下界
        ub = [0.00497,0.00249,0.00003288,0.8]  # 决策变量上界
        lbin = [1,1,1,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        L1 = Vars[:, [0]]
        L2 = Vars[:, [1]]
        C = Vars[:, [2]]
        R = Vars[:, [3]]


        f1 = 3/(80*L1)
        temp = (1-986960437.85*L2*C+986960437.85*C*C*R*R)**2+(986960437.85*L2*C*C*R)**2
        f2 = np.sqrt(temp) / ((1-986960437.85*L2*C)**2 + (C*C*R*R*986960437.85))
        f3 = ((L2*L2+986960647.76*L2*L2*C*C*R*R)/((L1+L2-986960647.76*L1*L2*C)**2+(986960647.76*C*C*R*R*(L1+L2)**2))) * \
             ((555165369990000*C*C*R)/(1+986960657.76*C*C*R*R)) + (4776880446.8*C*C*R)/(1+98695.877*C*C*R*R)
        f4 = L1 + L2
        f5 = L1 / L2

        pop.ObjV = np.hstack([f1, f2,f3, f4, f5])  # 把求得的目标函数值赋值给种群pop的ObjV


        # 约束条件拆开：
        CV1 = L1+L2-0.004972  # L1+L2-0.004972 <= 0
        CV2 = -L1-L2          # -L1-L2 <= 0
        CV3 = (L1+L2)/(L1*L2*C)-246740007.36  # (L1+L2)/(L1*L2*C)-246740007.36 <= 0
        CV4 = 9869587.73-(L1+L2)/(L1*L2*C)    # 9869587.73-(L1+L2)/(L1*L2*C) <= 0

        pop.CV = np.hstack([CV1, CV2, CV3, CV4])