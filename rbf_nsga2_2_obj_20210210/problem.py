import numpy as np
import pickle
import geatpy as ea


"""
决策变量的最小化目标双目标优化问题
min f1 
min f2 

s.t.(12个决策变量的上下界)
lower = [3.654,0.12,3.654,0.27,0,0.12,0,0.27,0.554,0.006,3.354,0.06]
upper = [4.054,0.2,4.054,0.3,0.4,0.2,0.4,0.3,0.754,0.1,3.554,0.1]
"""
# 读取训练好的rbf模型
with open("model1.pkl","rb") as f1:
    model1 = pickle.load(f1)

with open("model2.pkl", "rb") as f2:
    model2 = pickle.load(f2)

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'    # 初始化name（函数名称，可以随意设置）
        Dim = 12               # 初始化Dim（决策变量维数）
        maxormins = [1] * M   # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [3.654,0.12,3.654,0.27,0,0.12,0,0.27,0.554,0.006,3.354,0.06]       # 决策变量下界
        ub = [4.054,0.2,4.054,0.3,0.4,0.2,0.4,0.3,0.754,0.1,3.554,0.1]     # 决策变量上界
        lbin = [1] * Dim      # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim      # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        F2 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        for i in range(popsize):
            x1 = Vars[i, [0]][0]
            x2 = Vars[i, [1]][0]
            x3 = Vars[i, [2]][0]
            x4 = Vars[i, [3]][0]
            x5 = Vars[i, [4]][0]
            x6 = Vars[i, [5]][0]
            x7 = Vars[i, [6]][0]
            x8 = Vars[i, [7]][0]
            x9 = Vars[i, [8]][0]
            x10 = Vars[i, [9]][0]
            x11 = Vars[i, [10]][0]
            x12 = Vars[i, [11]][0]
            X = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]).reshape(1,12)

            # 计算目标函数值
            f1 = model1.predict(X)
            f2 = model2.predict(X)

            F1[i,0] = f1
            F2[i,0] = f2
        pop.ObjV = np.hstack([F1, F2])  # 把求得的目标函数值赋值给种群pop的ObjV
