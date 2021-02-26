import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f
min S

s.t.
0<= x1-x11 <= 2.5  流量
0<= x12-x22 <= 25  开始时间
0<= x23-x33 <= 25    结束时间
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 33  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        a = [2.5 for _ in range(11)]
        b = [25 for _ in range(22)]
        ub = a + b  # 决策变量上界
        lbin = [0] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        # 相关参数
        beta = 0.5
        Ai = 3.4
        mi = 0.5
        l = [10.23, 1.8, 4.2, 5.8, 1.25, 1.2, 0.88, 1.03, 1.4, 1.9, 6.17, 1.8]
        W = 280.5  # 预计来水量
        S = [46, 146, 248, 65, 76, 55, 41, 117, 230, 233, 53]  # 下级渠道面积
        q = [0.6, 1.0, 1.0, 0.6, 0.6, 0.5, 0.5, 0.6, 0.8, 1.5, 0.8]  # 下级渠道设计流量
        M = [0.12 * i for i in S]  # 综合灌水定额
        Q = 2.5  # 上级渠道设计流量
        SQ = 1259  # 上级渠道面积

        T = 25

        # 目标函数
        # todo 针对每一个个体，最后再合并
        f1 = []
        for j in range(len(Vars)): # 种群大小
            Quj = []
            for t in range(T):
                temp = 0
                for i in range(11):
                    a, b = abs(Vars[j, [i + 11]]), abs(Vars[j, [i + 22]])
                    if (a <= t <= b) or (b <= t <= a):
                        temp += q[i]
                Quj.append(temp)
            Qu_bar = sum(Quj) / T
            f1.append(sum([(val-Qu_bar)**2 for val in Quj]) / (T - 1))
        f1 = np.array(f1).reshape((len(Vars),1))

        temp_f2 = 0
        for i in range(11):
            period = abs(Vars[:, [i + 11]] - Vars[:, [i + 22]])
            temp_f2 += (beta * Ai * l[i] * (q[i] ** (1 - mi)) * period) / 100
        f2 = temp_f2

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV

        # ----------约束条件----------
        # 约束6
        constrain_6 = []
        for i in range(11):
            period = abs(Vars[:, [i + 11]] - Vars[:, [i + 22]])
            constrain_6.append(q[i] * period)
        CV1 = sum(constrain_6) - W

        # 约束7
        # todo  Mi是下级渠道控制区域下作物的灌溉定额
        CV2_0 = M * S[0] - q[0] * abs(Vars[:, [0 + 11]] - Vars[:, [0 + 22]])
        CV2_1 = M * S[1] - q[1] * abs(Vars[:, [1 + 11]] - Vars[:, [1 + 22]])
        CV2_2 = M * S[2] - q[2] * abs(Vars[:, [2 + 11]] - Vars[:, [2 + 22]])
        CV2_3 = M * S[3] - q[3] * abs(Vars[:, [3 + 11]] - Vars[:, [3 + 22]])
        CV2_4 = M * S[4] - q[4] * abs(Vars[:, [4 + 11]] - Vars[:, [4 + 22]])
        CV2_5 = M * S[5] - q[5] * abs(Vars[:, [5 + 11]] - Vars[:, [5 + 22]])
        CV2_6 = M * S[6] - q[6] * abs(Vars[:, [6 + 11]] - Vars[:, [6 + 22]])
        CV2_7 = M * S[7] - q[7] * abs(Vars[:, [7 + 11]] - Vars[:, [7 + 22]])
        CV2_8 = M * S[8] - q[8] * abs(Vars[:, [8 + 11]] - Vars[:, [8 + 22]])
        CV2_9 = M * S[9] - q[9] * abs(Vars[:, [9 + 11]] - Vars[:, [9 + 22]])
        CV2_10 = M * S[10] - q[10] * abs(Vars[:, [10 + 11]] - Vars[:, [10 + 22]])

        pop.CV = np.hstack([CV1, CV2_0, CV2_1, CV2_2, CV2_3, CV2_4, CV2_5, CV2_6, CV2_7, CV2_8, CV2_9, CV2_10])
