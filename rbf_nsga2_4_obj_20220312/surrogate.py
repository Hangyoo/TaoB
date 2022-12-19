import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm

"""
根据调参结果，下述参数用于构建rbf模型：
    Model1: 拟合f的rbf模型
    Model2: 拟合ha的rbf模型
    Model2: 拟合haz的rbf模型
    Model2: 拟合De的rbf模型
"""

warnings.filterwarnings("ignore")
# 读取数据
data = pd.read_excel("data.xls")

data_inputs = np.array(data.iloc[:,0:3])
# 读取数据标签(100*2)
f1 = np.array(data.iloc[:,3:4])
f2 = np.array(data.iloc[:,4:5])
f3 = np.array(data.iloc[:,4:5])
f4 = np.array(data.iloc[:,5:6])

ss_x = StandardScaler() # 实例化用于对特征标准化类
ss_y = StandardScaler() # 实例化用于对标签标准化类
# 对数据进行标准化
data_inputs = ss_x.fit_transform(pd.DataFrame(data_inputs))
# 读取数据标签
f1 = ss_y.fit_transform(pd.DataFrame(f1))
f2 = ss_y.fit_transform(pd.DataFrame(f2))
f3 = ss_y.fit_transform(pd.DataFrame(f3))
f4 = ss_y.fit_transform(pd.DataFrame(f4))

# Step1:第一个目标函数拟合
# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(data_inputs, f1, test_size=0.2,random_state=42)
# 随机森林模型训练
model1 = svm.SVR(kernel="rbf")
# 数据拟合
model1.fit(X_train, y_train)
# 模型保存
with open("model1.pkl", "wb") as f:
    pickle.dump(model1, f)


# Step2:第二个目标函数拟合
# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(data_inputs, f2, test_size=0.2,random_state=42)
# 随机森林模型训练
model2 = svm.SVR(kernel="rbf")
# 数据拟合
model2.fit(X_train, y_train)
# 模型保存
with open("model2.pkl", "wb") as f:
    pickle.dump(model2, f)

# Step3:第3个目标函数拟合
# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(data_inputs, f3, test_size=0.2,random_state=42)
# 随机森林模型训练
model3 = svm.SVR(kernel="rbf")
# 数据拟合
model3.fit(X_train, y_train)
# 模型保存
with open("model3.pkl", "wb") as f:
    pickle.dump(model3, f)


# Step4:第4个目标函数拟合
# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(data_inputs, f4, test_size=0.2,random_state=42)
# 随机森林模型训练
model4 = svm.SVR(kernel="rbf")
# 数据拟合
model4.fit(X_train, y_train)
# 模型保存
with open("model4.pkl", "wb") as f:
    pickle.dump(model4, f)