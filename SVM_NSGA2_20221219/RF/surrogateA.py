import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
# 读取数据
dataB = pd.read_excel("../data0.xlsx",sheet_name="低中高 新")

# 读取数据特征
data_inputs = np.array(dataB.iloc[:,1:11])
# 读取数据标签
f1 = np.array(dataB.iloc[:,11:12])
f2 = np.array(dataB.iloc[:,12:13])

ss_x = StandardScaler() # 实例化用于对特征标准化类
ss_y1 = StandardScaler() # 实例化用于对标签标准化类
ss_y2 = StandardScaler() # 实例化用于对标签标准化类

# 对数据进行标准化
data_inputs = ss_x.fit_transform(pd.DataFrame(data_inputs))
# 读取数据标签
f1 = ss_y1.fit_transform(pd.DataFrame(f1))
f2 = ss_y2.fit_transform(pd.DataFrame(f2))

X_train1, X_test1, y_train1, y_test1 = train_test_split(data_inputs, f1, test_size=0.1, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(data_inputs, f2, test_size=0.1, random_state=42)


# Step1:第一个目标函数拟合(Tavg)
model1 = RandomForestRegressor()
# 数据拟合
model1.fit(X_train1, y_train1)
# 模型保存
with open("modelB1.pkl", "wb") as f:
    pickle.dump(model1, f)


# Step2:第二个目标函数拟合(Trip)
model2 = RandomForestRegressor()
# 数据拟合
model2.fit(X_train2, y_train2)
# 模型保存
with open("modelB2.pkl", "wb") as f:
    pickle.dump(model2, f)


