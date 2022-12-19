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
dataB = pd.read_excel("./dataB.xls")

# 读取数据特征
data_inputs = np.array(dataB.iloc[:,3:8])
# 读取数据标签
f1 = np.array(dataB.iloc[:,1:2])
f2 = np.array(dataB.iloc[:,2:3])

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

# 对数据进行标准化
# data_inputs1 = ss_x.fit_transform(pd.DataFrame(X_train1))
# data_inputs2 = ss_x.fit_transform(pd.DataFrame(X_train2))

# 读取数据标签
# f1 = ss_y1.fit_transform(pd.DataFrame(y_train1))
# f2 = ss_y2.fit_transform(pd.DataFrame(y_train2))

# Step1:第一个目标函数拟合(Tavg)
# model1 = RandomForestRegressor()
model1 = SVR()
# 数据拟合
model1.fit(X_train1, y_train1)
# 模型保存
with open("modelB1.pkl", "wb") as f:
    pickle.dump(model1, f)


# Step2:第二个目标函数拟合(Trip)
# model2 = RandomForestRegressor()
model2 = SVR()
# 数据拟合
model2.fit(X_train2, y_train2)
# 模型保存
with open("modelB2.pkl", "wb") as f:
    pickle.dump(model2, f)

# R2(越大越好)
score1 = model1.score(X_test1,y_test1)
print(score1)

score2 = model2.score(X_test2,y_test2)
print(score2)


# y_pred1 = model1.predict(X_test1)
# plt.plot([i for i in range(len(y_test1))],y_test1,'o-')
# plt.plot([i for i in range(len(y_pred1))],y_pred1,'*-.')
# plt.title('Prediction on Test data',fontsize=13)
# plt.ylabel('$T_{avg}$',fontsize=13)
# plt.xlabel('Samples',fontsize=13)
# plt.show()
#
#
# y_pred2 = model2.predict(X_test2)
# plt.plot([i for i in range(len(y_test2))],y_test2,'o-')
# plt.plot([i for i in range(len(y_pred2))],y_pred2,'*-.')
# plt.title('Prediction on Test data',fontsize=13)
# plt.ylabel('$T_{rip}$',fontsize=13)
# plt.xlabel('Samples',fontsize=13)
# plt.show()


y_pred1 = ss_y1.inverse_transform(model1.predict(X_test1).reshape(-1,1))

# y_pred2 = ss_y2.inverse_transform(model2.predict(X_test2).reshape(-1,1))

data1 = np.concatenate((ss_x.inverse_transform(X_test1),ss_y1.inverse_transform(y_test1),y_pred1),axis=1)
# data1 = np.concatenate((ss_x.inverse_transform(X_test2),ss_y2.inverse_transform(y_test2),y_pred2),axis=1)
df = pd.DataFrame(data1,columns=['SMH', 'SMangle', 'STW2', 'Starc2',"EW2",'Tavg-true', 'Tavg-predict'])
# df = pd.DataFrame(data1,columns=['SMH', 'SMangle', 'STW2', 'Starc2',"EW2",'Trip-true', 'Trip-predict'])
df.to_excel('test_data_B1.xls')

