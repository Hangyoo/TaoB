import numpy as np
import pickle
import geatpy as ea


"""
决策变量的最小化目标双目标优化问题
max f1
max f2

s.t.
x1 叶片数量
x2 尾缘出口后弯角
x3 转速
x4 出口直径
x5 轮缘直径
"""

# 读取训练好的机器学习模型
with open("Bag_model1.pkl","rb") as f1:
    model1 = pickle.load(f1)

with open("Bag_model2.pkl", "rb") as f2:
    model2 = pickle.load(f2)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'    # 初始化name（函数名称，可以随意设置）
        Dim = 5               # 初始化Dim（决策变量维数）
        maxormins = [-1,-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1,0,1,0,0]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [8,40,100000,20,16]       # 决策变量下界
        ub = [18,65,300000,30,25]        # 决策变量上界
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
            X = np.array([x1,x2,x3,x4,x5]).reshape(1,5)
            # 计算目标函数值
            f1 = model1.predict(X)
            f2 = model2.predict(X)
            F1[i,0] = f1
            F2[i,0] = f2
        pop.ObjV = np.hstack([F1, F2])  # 把求得的目标函数值赋值给种群pop的ObjV
