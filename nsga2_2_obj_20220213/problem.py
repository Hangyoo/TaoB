import numpy as np
import geatpy as ea
import pickle
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

"""
最小(大)化目标双目标优化问题
max f1: NPV 
min f2：碳排放

"""

M = 1
r = 0.1
P = 0.73
P1 = 2.58e-8

# 读取训练好的机器学习模型
with open("model.pkl","rb") as f1:
    model1 = pickle.load(f1)

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2): # 目标函数为2
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 9  # 初始化Dim（决策变量维数）
        maxormins = [-1,1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1,1,1,1,1,1,0,0,0]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0,0,0,0,0,0,3,0.2,0.2]  # 决策变量下界
        ub = [35,35,35,35,35,35,6.5,0.9,0.9]  # 决策变量上界
        lbin = [1,1,1,1,1,1,1,1,1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1,1,1,1,1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.df = pd.read_excel("data.xlsx",header=0,skiprows=[1],usecols=range(2,12),dtype=float)
        self.scaler_x = preprocessing.StandardScaler()
        self.scaler_xx = self.scaler_x.fit_transform(self.df[self.df.columns[:-1]])
        self.scaler_y = preprocessing.StandardScaler()
        self.scaler_yy = self.scaler_y.fit_transform(self.df[[self.df.columns[-1]]])
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        popsize = Vars.shape[0]

        F1 = np.array([float('-Inf')] * popsize).reshape(popsize, 1)
        F2 = np.array([float('-Inf')] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            x1 = Vars[i, [0]][0] # 东墙PCM
            x2 = Vars[i, [1]][0] # 西墙PCM
            x3 = Vars[i, [2]][0] # 南墙PCM
            x4 = Vars[i, [3]][0] # 北墙PCM
            x5 = Vars[i, [4]][0] # 屋顶PCM
            x6 = Vars[i, [5]][0] # 楼板PCM
            x7 = Vars[i, [6]][0] # 窗外U值
            x8 = Vars[i, [7]][0] # 窗墙比
            x9 = Vars[i, [8]][0] # SHGC

            # 对变量进行初始排序
            temp = [x1, x2, x3, x4, x5, x6]
            temp_sort = sorted(temp,reverse=True)
            x3, x1, x2, x5, x4, x6 = temp_sort[0],temp_sort[1],temp_sort[2],temp_sort[3],temp_sort[4],temp_sort[5]

            # 判断约束条件，对违反约束的解进行惩罚
            punish = 0
            if x3 < x1 or x1 < x2 or x2 < x5 or x5 < x4 or x4 < x6:
                print('100')
                punish = 1e15

            # 根据ANN模型计算E
            X = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9]).reshape(1, 9)
            # 数据处理
            X = np.array(self.scaler_x.transform(X))
            E = self.scaler_y.inverse_transform(model1.predict(X).reshape(-1,1))
            Q = 3118.59 - E
            I = 30
            # 两个目标函数
            f1 = sum([(Q * P) / ((1 + r) ** i) for i in range(31)]) - P1 * (3.5e7 * x1 + 3.5e7 * x2 + 3.5e7 * (1 - x8) * x3 + 3.5e7 * x4 * 2.5e7 * x5 + 2.5e7 * x6)
            f2 = M * E * I
            F1[i, 0] = f1 + punish
            F2[i, 0] = f2 - punish

        pop.ObjV = np.hstack([F1, F2])  # 把求得的目标函数值赋值给种群pop的ObjV





