import numpy as np
import pickle
import geatpy as ea


"""
最大化单目标的优化问题
max f1
s.t.
x1 ∈ [50,200]
x2 ∈ [10,30]
x3 ∈ [5,25]
"""

# 读取训练好的机器学习模型
with open("model.pkl","rb") as f1:
    model1 = pickle.load(f1)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'    # 初始化name（函数名称，可以随意设置）
        Dim = 3               # 初始化Dim（决策变量维数）
        maxormins = [-1] * M   # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [50,10,5]       # 决策变量下界
        ub = [200,30,25]     # 决策变量上界
        lbin = [1] * Dim      # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim      # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        for i in range(popsize):
            x1 = Vars[i, [0]][0]
            x2 = Vars[i, [1]][0]
            x3 = Vars[i, [2]][0]
            X = np.array([x1,x2,x3]).reshape(1,3)
            # 计算目标函数值
            f1 = model1.NN_predict(X)
            F1[i,0] = f1
        pop.ObjV = F1  # 把求得的目标函数值赋值给种群pop的ObjV
