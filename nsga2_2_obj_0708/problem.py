import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 时间
min f2 质量误差

s.t.
n 500<= x1 <= 1500
v 10<= x2 <= 35
16ms<= x3 <= 24
"""

V = 53470.0
O = 0.2
alpha = 71 * np.pi / 180
a = 42.22
H = 30.0
T1 = 0.17

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 2  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [760,4.8]  # 决策变量下界
        ub = [2280,11.3]  # 决策变量上界
        lbin = [1,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        x1 = Vars[:, [0]]  # n
        x2 = Vars[:, [1]]  # v
        b = (2.6725 + 0.001082 * x1 - 0.09283 * x2 + 0.001292 * x2 * x2 - 0.00001 * x1 * x2) ** 2
        h = (1.1942 + 0.000297 * x1 - 0.0055 * x2 + 0.000262 * x2 * x2 - 0.000008 * x1 * x2) ** 2

        f1 = V / h / b / (1 - O) / x2 + a * T1 * H / h / b / (1 - O)

        f2 =  (h ** 2) / 2 / np.tan(alpha)


        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV