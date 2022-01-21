import numpy as np
import geatpy as ea
import pickle

"""
最小化目标双目标优化问题
min f1 = O1
min f2 = O2

s.t.
0.3 <= x1 <= 0.6
0.5 <= x2 <= 5.5
1e8 <= x3 <= 1e9
3.0 <= x4 <= 5.0
2600 <= x5 <= 2700
"""

# 读取训练好的机器学习模型
with open("model1.pkl","rb") as f:
    model1 = pickle.load(f)

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 5  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0,0,0,0,1]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.3,0.5,1e8,3.0,2600]  # 决策变量下界
        ub = [0.6,5.5,1e9,5.0,2700]  # 决策变量上界
        lbin = [1,1,1,1,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        popsize = Vars.shape[0]

        # 计算目标函数f1
        F1 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            x1 = Vars[i, [0]][0]
            x2 = Vars[i, [1]][0]
            X = np.array([x1,x2]).reshape(1,2)
            f1 = model1.predict(X)
            F1[i,0] = abs(f1 - 33.1) / 33.1


        # 计算目标函数f2
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        x5 = np.array(Vars[:, [4]],dtype=int)

        F2 = (x4-3.0)/(5.0-3.0) - (x5-2600)/(2700-2600) + (x3-1e8)/(1e9-1e8) + 2

        pop.ObjV = np.hstack([F1, F2])  # 把求得的目标函数值赋值给种群pop的ObjV


if __name__ == "__main__":
    with open("model2.pkl", "rb") as f:
        model1 = pickle.load(f)
    x1 = 0.3
    x2 = 0.5
    X = np.array([x1, x2]).reshape(1, 2)
    f1 = model1.predict(X)
    print(f1)