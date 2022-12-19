import numpy as np
import geatpy as ea
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc
import warnings
import pandas as pd
import numpy as np
import random


np.random.seed(100)
warnings.filterwarnings("ignore")

# 加载数据集
data = pd.read_csv("./data.csv")
X = np.array(data.iloc[:,:256])   # 读取数据集特征
y = np.array(data.iloc[:,-1]) # 读取数据集标签
print(f'数据集中包含的类别分别为(3分类问题)：',set(y))

# 对数据进行标准化
ss_x = StandardScaler()
X = ss_x.fit_transform(pd.DataFrame(X))

# 样本数， 特征数
n_samples, n_features = X.shape

# 训练样本的类别数量
n_classes = 3

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""
最大化准确率Acc的单目标的优化问题
max acc

优化变量：
1. 核函数 kernel (x1)
2. C值 (x2)
3. gamma值 (x3)
4. degree值  (x4)
5. decision_function_shape 选择聚合的方式，ovo 还是 ovr  x(5)

s.t.
x1 ∈ ['linear'，'poly'，'rbf'，'sigmoid'，'precomputed']
x2 ∈ [0.1,10]
x3 ∈ [0.1,2] 
x4 ∈ [0.1,10]
x5 ∈ ['ovo','ovr'] # 1对1 或 1对多
"""

KERNEL = ['linear','poly','rbf','sigmoid','precomputed']
MODEL = ['ovo','ovr']

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 5  # 初始化Dim（决策变量维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1,0,0,1,1]   # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0, 0.1, 0.1, 0.1, 0]  # 决策变量下界
        ub = [3, 10, 2, 10, 1]  # 决策变量上界
        lbin = [1, 1, 1, 1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1, 1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)
        popsize = Vars.shape[0]

        F1 = np.array([float('-Inf')] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            x1 = int(Vars[i, [0]][0]) # 核函数 kernel
            x2 = Vars[i, [1]][0] # C值
            x3 = Vars[i, [2]][0] # gamma值
            x4 = Vars[i, [3]][0] # degree值
            x5 = int(Vars[i, [4]][0]) # decision_function_shape 选择聚合的方式，ovo 还是 ovr
            clf = svm.SVC(kernel=KERNEL[x1], C=x2, gamma=x3, degree=x4, decision_function_shape=MODEL[x5],random_state=0)
            # 在训练集上训练模型
            clf.fit(X_train, y_train)
            # 在测试集上测试模型
            acc_score = clf.score(X_test, y_test)
            del clf
            gc.collect()
            F1[i, 0] = acc_score
        pop.ObjV = np.hstack([F1])  # 把求得的目标函数值赋值给种群pop的ObjV
