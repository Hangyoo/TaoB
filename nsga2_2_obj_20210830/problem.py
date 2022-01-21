import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 = MRR 
min f2 = Ra

s.t.
0.6 <= a <= 1
8 <= b <= 12
0.5 <= c <= 0.7
1000 <= d <= 1200
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 4  # 初始化Dim（决策变量维数）
        maxormins = [1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.6, 8, 0.5, 1000]  # 决策变量下界
        ub = [1, 12, 0.7, 1200]  # 决策变量上界
        lbin = [1,1,1,1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        a = Vars[:, [0]]
        b = Vars[:, [1]]
        c = Vars[:, [2]]
        d = Vars[:, [3]]

        f1 = 3.3*d - 460.0*b + 281.5*a*b - 2.229*a*d - 129.3*b*c + 0.425*b*d + 1.498*c*d - 7.883*(b**2) - 0.002712*(d**2)

        f2 = 0.01744*b - 0.0004161*d + 1.045*a*b - 0.009339*a*d - 1.688*b*c + 0.001683*b*d + 0.01633*c*d - 0.08413*(b**2) - 7.981e-6*(d**2)

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV
