import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

#---------------数据预处理------------------------
data = pd.read_excel(r"C:\Users\Hangyu\Desktop\Data.xls")

# 读取数据特征(144*3)
data_inputs = np.array(data.iloc[:,0:3])

# 读取数据标签(144*1)
data_outputs = np.array(data.iloc[:,4:5])

# 数据标准化
mm = MinMaxScaler()
labels = mm.fit_transform(data_outputs)

# Machine-learning model training

# Step1 首先对 n_estimators 进行调参 (调参显示最优n_estimators=35)
# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_outputs, test_size=0.7,random_state=42)
Mse,R2 = [],[]
for i in range(10,200,5):
    clf = RandomForestRegressor(n_estimators=i, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    Mse.append(mse)
    R2.append(r2)
print(f"MSE指标最小时,n_estimators={range(10,200,5)[Mse.index(min(Mse))]}")
print(f"R2指标最大时,n_estimators={range(10,200,5)[R2.index(max(R2))]}")

# MSE值绘制,越小越好
plt.plot(range(10,200,5),Mse)
plt.xlabel("n_estimators")
plt.ylabel("MSE")
plt.show()

# R2值绘制,越小越好
plt.plot(range(10,200,5),R2)
plt.xlabel("n_estimators")
plt.ylabel("R2")
plt.show()

# Step2 其次对 max_depth 进行调参 (调参显示最优max_depth=7)
# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_outputs, test_size=0.7,random_state=42)
Mse,R2 = [],[]
for i in range(1,20):
    clf = RandomForestRegressor(n_estimators=95, random_state=0,max_depth=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    Mse.append(mse)
    R2.append(r2)
print(f"MSE指标最小时,max_depth={range(1,20)[Mse.index(min(Mse))]}")
print(f"R2指标最大时,max_depth={range(1,20)[R2.index(max(R2))]}")

# MSE值绘制,越小越好
plt.plot(range(1,20),Mse)
plt.xlabel("n_estimators")
plt.ylabel("MSE")
plt.show()

# R2值绘制,越小越好
plt.plot(range(1,20),R2)
plt.xlabel("n_estimators")
plt.ylabel("R2")
plt.show()




