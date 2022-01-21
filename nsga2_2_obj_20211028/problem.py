import pandas as pd
import warnings
import numpy as np
import geatpy as ea


"""
最小化目标双目标优化问题
min f1 = TN
min f2 = TP
min f3 = COST
"""

warnings.filterwarnings("ignore")

# 读取数据
data1 = pd.read_excel("./data.xlsx",sheet_name='TN')
data1 = data1.drop(columns=['Cell_ID'],axis=1)
data1.set_axis([i for i in range(data1.shape[1])],axis="columns",inplace=True)

# 读取数据
data2 = pd.read_excel("./data.xlsx",sheet_name='TP')
data2 = data2.drop(columns=['Cell_ID'],axis=1)
data2.set_axis([i for i in range(data2.shape[1])],axis="columns",inplace=True)

# 读取数据
data3 = pd.read_excel("./data.xlsx",sheet_name='COST')
data3 = data3.drop(columns=['Cell_ID'],axis=1)
data3.set_axis([i for i in range(data3.shape[1])],axis="columns",inplace=True)



class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 44  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0]*Dim  # 决策变量下界
        ub = [12]*Dim  # 决策变量上界
        lbin = [1]*Dim   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F2 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F3 = np.array([float("-Inf")] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            individual = []
            f1 = 0
            f2 = 0
            f3 = 0
            for j in range(44):
                individual.append(Vars[i, [j]][0])
            for j in range(44):
                gene = individual[j]
                f1 += data1.iloc[j,gene]
                f2 += data2.iloc[j,gene]
                f3 += data3.iloc[j,gene]
            # print(f1,f2,f3)
            # print(i,individual)
            F1[i, 0] = round(f1,3)   # TN
            F2[i, 0] = round(f2,3)   # TP
            F3[i, 0] = round(f3,3)   # COST

        pop.ObjV = np.hstack([F1, F2, F3])  # 把求得的目标函数值赋值给种群pop的ObjV