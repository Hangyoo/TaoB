import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle


warnings.filterwarnings("ignore")
# 读取数据
data = pd.read_csv("./data.csv",header=None,index_col=0)

# 读取数据特征（前两行特征用处不大，不作为训练模型的特征考虑）
data_inputs = np.array(data.iloc[:,3:])
# data_inputs = np.array(pd.concat([data.iloc[:,0:2],data.iloc[:,3:]],axis=1))  # 想用前两个特征可以使用这行代码

# 读取数据标签
close = np.array(data.iloc[:,2])

ss_x = StandardScaler() # 实例化用于对特征标准化类
ss_y = StandardScaler() # 实例化用于对标签标准化类
# 对数据进行标准化
data_inputs = ss_x.fit_transform(pd.DataFrame(data_inputs))
# 读取数据标签
close = ss_y.fit_transform(pd.DataFrame(close))

# 目标函数拟合
# SVM模型训练
model1 = SVR(kernel='rbf',C=1.0,gamma=0.01)
# 数据拟合
model1.fit(data_inputs, close)

# 模型保存
with open("SVR1.pkl", "wb") as f:
    pickle.dump(model1, f)

# 输出模型的MSE 均方误差
close_pred = model1.predict(data_inputs)
MSE = mean_squared_error(close_pred,close)
print('MSE:',MSE)

# 绘制图像
plt.plot([i for i in range(len(close))],close)  # 真实值
plt.plot([i for i in range(len(close_pred))],close_pred)  # 真实值
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(['True','Predict'])
plt.show()