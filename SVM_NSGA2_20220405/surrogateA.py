import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
# 读取数据
dataA = pd.read_excel("./dataA.xls")

# 读取数据特征
data_inputs = np.array(dataA.iloc[:,3:7])
# 读取数据标签
f1 = np.array(dataA.iloc[:,1:2])
f2 = np.array(dataA.iloc[:,2:3])


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
# model1 = RandomForestRegressor()
model1 = SVR()
# 数据拟合
model1.fit(X_train1, y_train1)
# 模型保存
with open("modelA1.pkl", "wb") as f:
    pickle.dump(model1, f)


# Step2:第二个目标函数拟合(Trip)
# model2 = RandomForestRegressor()
model2 = SVR()
# 数据拟合
model2.fit(data_inputs, f2)
# 模型保存
with open("modelA2.pkl", "wb") as f:
    pickle.dump(model2, f)


y_pred1 = ss_y1.inverse_transform(model1.predict(X_test1).reshape(-1,1))
# plt.plot([i for i in range(len(y_test1))],y_test1,'o-')
# plt.plot([i for i in range(len(y_pred1))],y_pred1,'*-.')
# plt.title('Prediction on Test data',fontsize=13)
# plt.ylabel('$T_{avg}$',fontsize=13)
# plt.xlabel('Samples',fontsize=13)
# plt.show()
#
#
y_pred2 = ss_y2.inverse_transform(model2.predict(X_test2).reshape(-1,1))
# plt.plot([i for i in range(len(y_test2))],y_test2,'o-')
# plt.plot([i for i in range(len(y_pred2))],y_pred2,'*-.')
# plt.title('Prediction on Test data',fontsize=13)
# plt.ylabel('$T_{rip}$',fontsize=13)
# plt.xlabel('Samples',fontsize=13)
# plt.show()

data1 = np.concatenate((ss_x.inverse_transform(X_test1),ss_y1.inverse_transform(y_test1),y_pred1),axis=1)
# data1 = np.concatenate((ss_x.inverse_transform(X_test2),ss_y2.inverse_transform(y_test2),y_pred2),axis=1)
df = pd.DataFrame(data1,columns=['STW1', 'Starc1', 'EW1', 'Splitratio','Tavg-true', 'Tavg-predict'])
# df = pd.DataFrame(data1,columns=['STW1', 'Starc1', 'EW1', 'Splitratio','Trip-true', 'Trip-predict'])
df.to_excel('test_data_A1.xls')

