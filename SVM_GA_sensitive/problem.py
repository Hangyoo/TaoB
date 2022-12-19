import numpy as np
import geatpy as ea
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import gc
import warnings
import pandas as pd
import numpy as np
import random
from sklearn import metrics
from xgboost import XGBRegressor


np.random.seed(100)
warnings.filterwarnings("ignore")

# 加载数据集
datas = np.genfromtxt('wind_pressure.csv', delimiter=',')
X = datas[:, :4]
y = datas[:, 4]

ss_x = StandardScaler()
X = ss_x.fit_transform(X)

ss_y = StandardScaler()
y = ss_y.fit_transform(y.reshape(-1,1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=30)

# 样本数， 特征数
n_samples, n_features = X.shape


"""
最大化R2， 最小化mse

s.t.
x1 ∈ [0,1,...,200]
x2 ∈ [1,...,25]
x3 ∈ [0,1] 
x4 ∈ [1,...,20]
x5 ∈ [0,1] 
x6 ∈ [0,1] 
"""

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 6  # 初始化Dim（决策变量维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [ 1 ,1, 0, 1, 0, 0]   # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0, 1, 0, 1, 0, 0]  # 决策变量下界
        ub = [200, 25, 1, 20, 1, 1]  # 决策变量上界
        lbin = [0, 1, 0, 1, 0, 0]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 0, 1, 0, 0]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        popsize = Vars.shape[0]

        F1 = np.array([float('-Inf')] * popsize).reshape(popsize, 1)

        for i in range(popsize):
            x1 = int(Vars[i, [0]][0]) # n_estimators
            x2 = int(Vars[i, [1]][0]) # max_depth
            x3 = Vars[i, [2]][0] # learning_rate
            x4 = int(Vars[i, [3]][0]) # min_child_weight
            x5 = Vars[i, [4]][0] # subsample
            x6 = Vars[i, [5]][0] # reg_lambda

            # XGBOOST
            xgb_model = XGBRegressor(n_estimators=x1,
                                     max_depth=x2,
                                     learning_rate=x3,
                                     min_child_weight=x4,
                                     subsample=x5,
                                     reg_lambda=x6)

            # xgb_model = XGBRegressor()

            # SVR
            svr = SVR()
            svr.fit(X_train, y_train)
            print("SVR的R2:",svr.score(X_train,y_train))

            # 在训练集上训练模型
            xgb_model.fit(X_train, y_train)
            print("XGBOOST的R2:",xgb_model.score(X_train,y_train))
            # 在测试集上测试模型
            prediction = xgb_model.predict(X_test)
            r2 = metrics.r2_score(y_test, prediction)
            mse = metrics.mean_squared_error(y_test, prediction)
            F1[i, 0] = 1 - r2 + mse
            print("变量值:",x1,x2,x3,x4,x5,x6,"目标值:r2:",r2,"mse:",mse)
            # 释放内存
            del xgb_model
            gc.collect()
        pop.ObjV = np.hstack([F1])  # 把求得的目标函数值赋值给种群pop的ObjV
