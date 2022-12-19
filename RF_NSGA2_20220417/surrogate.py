import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
# 读取数据
dataB = pd.read_excel("./data.xls")

# 读取数据特征
data_inputs = np.array(dataB.iloc[:,:9])
# 读取数据标签
f1 = np.array(dataB.iloc[:,9:10])
f2 = np.array(dataB.iloc[:,10:11])

X_train, X_test, y_train, y_test = train_test_split(data_inputs, f1, test_size=0.1, random_state=42)

ss_x = StandardScaler() # 实例化用于对特征标准化类
ss_y1 = StandardScaler() # 实例化用于对标签标准化类
ss_y2 = StandardScaler() # 实例化用于对标签标准化类

# 对数据进行标准化
data_inputs = ss_x.fit_transform(pd.DataFrame(data_inputs))

# 读取数据标签
f1 = ss_y1.fit_transform(pd.DataFrame(f1))
f2 = ss_y2.fit_transform(pd.DataFrame(f2))

# Step1:第一个目标函数拟合(Tavg)
model1 = RandomForestRegressor()
# 数据拟合
model1.fit(data_inputs, f1)
# 模型保存
with open("model1.pkl", "wb") as f:
    pickle.dump(model1, f)


# Step2:第二个目标函数拟合(Trip)
model2 = RandomForestRegressor()
# 数据拟合
model2.fit(data_inputs, f2)
# 模型保存
with open("model2.pkl", "wb") as f:
    pickle.dump(model2, f)


