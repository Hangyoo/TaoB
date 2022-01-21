import numpy as np
import pickle
import geatpy as ea
# from sklearn.externals import joblib
import pickle


"""
最大化单目标的优化问题
max f1
s.t.
x1 ∈ [0.2,2]
x2 ∈ [0.3,2.7]
x3 ∈ [0.4,3.8]
"""

# 读取训练好的机器学习模型
with open("model.pkl","rb") as f:
    model_A = pickle.load(f)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'    # 初始化name（函数名称，可以随意设置）
        Dim = 3               # 初始化Dim（决策变量维数）
        maxormins = [-1] * M   # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.2,0.3,0.4]       # 决策变量下界
        ub = [2,2.7,3.8]     # 决策变量上界
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
            f1 = 0

            # 求x4在[37.5,60]的区间下，y的总和
            for x4 in np.arange(37.5,60.01,0.1): # 若要包含x4=60的点，需要将区间设置为60.01
                X = np.array([x1,x2,x3,x4]).reshape(1,4)
                # 计算目标函数值
                f1 += model_A.NN_predict(X)
            F1[i,0] = f1
        pop.ObjV = F1  # 把求得的目标函数值赋值给种群pop的ObjV
