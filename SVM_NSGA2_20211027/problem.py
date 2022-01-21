import numpy as np
import pickle
import geatpy as ea


"""
决策变量的最小化目标双目标优化问题
min f1 粗糙度
min f2 过切率
max f3 切深

s.t.
x1 ∈ [0,5] # 进给率
x2 ∈ [0.01,0.3] # 振幅
"""
# 读取训练好的机器学习模型
with open("model1.pkl","rb") as f1:
    model1 = pickle.load(f1)

with open("model2.pkl", "rb") as f2:
    model2 = pickle.load(f2)

with open("model3.pkl", "rb") as f3:
    model3 = pickle.load(f3)

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'    # 初始化name（函数名称，可以随意设置）
        Dim = 2               # 初始化Dim（决策变量维数）
        maxormins = [1,1,-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.0,0.01]       # 决策变量下界
        ub = [5.0,0.3]        # 决策变量上界
        lbin = [1] * Dim      # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim      # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        # todo 修改下面代码
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        F2 = np.array([float("Inf")]*popsize).reshape(popsize,1)
        F3 = np.array([float("-Inf")] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            x1 = Vars[i, [0]][0]
            x2 = Vars[i, [1]][0]
            X = np.array([x1,x2]).reshape(1,2)
            # 计算目标函数值
            f1 = model1.predict(X)
            f2 = model2.predict(X)
            f3 = model3.predict(X)
            F1[i,0] = f1
            F2[i,0] = f2
            F3[i,0] = f3
        pop.ObjV = np.hstack([F1, F2, F3])  # 把求得的目标函数值赋值给种群pop的ObjV
