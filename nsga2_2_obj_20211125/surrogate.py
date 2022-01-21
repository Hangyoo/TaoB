import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
# 读取数据
data = pd.read_excel("./data.xls")

# 读取数据特征
data_inputs = np.array(data.iloc[:,0:2])
# 读取数据标签
f1 = np.array(data.iloc[:,2:3])

# 目标函数f1拟合

# SVM模型训练
model1 = SVR()
# 数据拟合
model1.fit(data_inputs, f1)
# 模型保存
with open("model1.pkl", "wb") as f:
    pickle.dump(model1, f)

# SVM模型训练
model2 = RandomForestRegressor(n_estimators=100)
# 数据拟合
model2.fit(data_inputs, f1)
# 模型保存
with open("model2.pkl", "wb") as f:
    pickle.dump(model2, f)
