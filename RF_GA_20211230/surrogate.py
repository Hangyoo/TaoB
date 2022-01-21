import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

def norm(x):
    return (x - x.min(axis = 0)) / (x.max(axis = 0) - x.min(axis = 0))

# 读取数据
dataset = pd.read_excel(r"./data.xlsx",sheet_name='Sheet1')

ohe = OrdinalEncoder()
dataset_transform = ohe.fit_transform(dataset.iloc[:,2:4])
dataset.iloc[:,2:4] = dataset_transform
rop = dataset.pop('机械钻速')
# 数据归一化
norm_dataset = norm(dataset)

X_train,X_test,y_train,y_test = train_test_split(dataset,rop,random_state = 6,test_size=0.40)
print(X_train.head())
rf = RandomForestRegressor(n_estimators=100,random_state=60).fit(X_train, y_train)
predictions_train= rf.predict(X_train)
predictions_test=rf.predict(X_test)
print("RF训练集得分:{}".format(rf.score(X_train, y_train)))
print("RF测试集得分:{}".format(rf.score(X_test,y_test)))


# depth = 1
# x1,x2,x3,x4,x5,x6,x7,x8 = 0,0.015551,0.254310,0.004158,0.771739,0.434783,0.091575,0.231420
#
# X = np.array([depth, x1, 0.333333, 0.285714, x2, x3, x4, x5, x6,x7,x8]).reshape(1, 11)
# # 计算目标函数值
# f1 = rf.predict(X)  # loss
# print(f1)


with open("RF.pkl", "wb") as f:
    pickle.dump(rf, f)







