import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

"""
根据调参结果，下述参数用于构建随机深林模型：
    Model1: n_estimators=100; max_depth=8
"""

warnings.filterwarnings("ignore")
# 读取数据
data = pd.read_excel("Data.xls")
# 读取数据特征(9*3)
data_inputs = np.array(data.iloc[:,0:3])
# 读取数据标签(144*2)
f1 = np.array(data.iloc[:,3:4])

# Step1:第一个目标函数拟合
# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(data_inputs, f1, test_size=0.7,random_state=42)
# 随机森林模型训练
model = RandomForestRegressor(n_estimators=100,random_state=0,max_depth=3)

# 采用支持向量机回归
#model = SVR()

# 数据拟合
model.fit(X_train, y_train)
# 模型保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

