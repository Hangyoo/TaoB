import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

warnings.filterwarnings("ignore")
# 读取数据
data = pd.read_excel("./data.xls")

# 读取数据特征
data_inputs = np.array(data.iloc[:,0:5])
# 读取数据标签
f1 = np.array(data.iloc[:,5:6])
f2 = np.array(data.iloc[:,6:7])

# Step1:第一个目标函数拟合
# SVM模型训练
# model1 = SVR()
# 随机深林模型训练
# model1 = RandomForestRegressor(n_estimators=100)
# Bagging
model1 = BaggingRegressor()
# 数据拟合
model1.fit(data_inputs, f1)
# 模型保存
with open("Bag_model1.pkl", "wb") as f:
    pickle.dump(model1, f)


# Step2:第二个目标函数拟合
# SVM模型训练
# model2 = SVR()
# 随机深林模型训练
# model2 = RandomForestRegressor(n_estimators=100)
# Bagging
model2 = BaggingRegressor()
# 数据拟合
model2.fit(data_inputs, f2)
# 模型保存
with open("Bag_model2.pkl", "wb") as f:
    pickle.dump(model2, f)


