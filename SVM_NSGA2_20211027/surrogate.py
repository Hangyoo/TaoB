import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
# 读取数据
data = pd.read_excel("./data.xls")

# 读取数据特征
data_inputs = np.array(data.iloc[:,0:2])
# 读取数据标签
f1 = np.array(data.iloc[:,2:3])
f2 = np.array(data.iloc[:,3:4])
f3 = np.array(data.iloc[:,4:5])

# Step1:第一个目标函数拟合
# 验证集 测试集 数据划分
# X_train, X_test, y_train, y_test = train_test_split(data_inputs, f1, test_size=0.7,random_state=42)
# SVM模型训练
model1 = SVR()
# 数据拟合
model1.fit(data_inputs, f1)
# 模型保存
with open("model1.pkl", "wb") as f:
    pickle.dump(model1, f)


# Step2:第二个目标函数拟合
# 验证集 测试集 数据划分
# X_train, X_test, y_train, y_test = train_test_split(data_inputs, f2, test_size=0.7,random_state=42)
# SVM模型训练
model2 = SVR()
# 数据拟合
model2.fit(data_inputs, f2)
# 模型保存
with open("model2.pkl", "wb") as f:
    pickle.dump(model2, f)


# Step3:第三个目标函数拟合
# 验证集 测试集 数据划分
# X_train, X_test, y_train, y_test = train_test_split(data_inputs, f3, test_size=0.7,random_state=42)
# SVM模型训练
model3 = SVR()
# 数据拟合
model3.fit(data_inputs, f3)
# 模型保存
with open("model3.pkl", "wb") as f:
    pickle.dump(model3, f)
