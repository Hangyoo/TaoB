import numpy as np
import pandas as pd
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 
max f2
max f3

s.t.
0 <= x1 <= 100 (As)
0 <= x2 <= 100 (Ab)
0 <= x3 <= 100 (Az)
"""

# 数据读取
R = pd.read_excel(r"C:\Users\DELL\PycharmProjects\TB\nsga2_2_obj_20210201\匹配度.xls").iloc[:,1:]  # 匹配度矩阵
R.columns = [*map(lambda x:str(x), [i for i in range(20)])]
M = pd.read_excel(r"C:\Users\DELL\PycharmProjects\TB\nsga2_2_obj_20210201\特征值.xlsx").iloc[:,1:] # 特征值
M.columns = ["0","1","2"]
T = pd.read_excel(r"C:\Users\DELL\PycharmProjects\TB\nsga2_2_obj_20210201\等级.xlsx",header=None).iloc[:,1:]   # 等级
# print(R.iloc[3,4])

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 20  # 初始化Dim（决策变量维数）
        maxormins = [1,-1,-1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim # 决策变量下界
        ub = [1]* Dim  # 决策变量上界
        lbin = [1]* Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]* Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        E = [70,70,70] # 期望特征值

        f1 = []
        for pop_idx in range(len(Vars)):
            f1_temp = 0
            for i in range(20):
                sum = 0
                for j in range(3):
                    sum += (M.iloc[i,j] - E[j])
                f1_temp += Vars[pop_idx, [i]] * sum
            f1.append(f1_temp)
        f1 = np.array(f1).reshape(len(Vars),1)

        f2 = []
        for pop_idx in range(len(Vars)):
            f2_temp = 0
            for i in range(20):
                for j in range(20):
                    f2_temp += Vars[pop_idx, [i]] * R.iloc[i,j] #匹配度
            f2.append(f2_temp)
        f2 = np.array(f2).reshape(len(Vars), 1)

        f3 = []
        for pop_idx in range(len(Vars)):
            f3_temp = 0
            for i in range(20):
                f3_temp += Vars[pop_idx, [i]] * T.iloc[i,0]
            f3.append(f3_temp)
        f3 = np.array(f3).reshape(len(Vars), 1)

        pop.ObjV = np.hstack([f1, f2, f3])  # 把求得的目标函数值赋值给种群pop的ObjV

        # 约束处理
        CV1 = Vars[:, [0]] + Vars[:, [1]] + Vars[:, [2]]+ Vars[:, [3]] + Vars[:, [4]] - 3
        CV2 = Vars[:, [5]] + Vars[:, [6]] + Vars[:, [7]]+ Vars[:, [8]] + Vars[:, [9]] - 3
        CV3 = Vars[:, [10]] + Vars[:, [11]] + Vars[:, [12]]+ Vars[:, [13]] + Vars[:, [14]] - 2
        CV4 = Vars[:, [15]] + Vars[:, [16]] + Vars[:, [17]]+ Vars[:, [18]] + Vars[:, [19]] - 2
        CV5 = 3 - (Vars[:, [0]] + Vars[:, [1]] + Vars[:, [2]] + Vars[:, [3]] + Vars[:, [4]])
        CV6 = 3 - (Vars[:, [5]] + Vars[:, [6]] + Vars[:, [7]] + Vars[:, [8]] + Vars[:, [9]])
        CV7 = 2 - (Vars[:, [10]] + Vars[:, [11]] + Vars[:, [12]] + Vars[:, [13]] + Vars[:, [14]])
        CV8 = 2 - (Vars[:, [15]] + Vars[:, [16]] + Vars[:, [17]] + Vars[:, [18]] + Vars[:, [19]])
        pop.CV = np.hstack([CV1, CV2, CV3, CV4,CV5, CV6, CV7, CV8])
