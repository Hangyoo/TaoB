import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

data = pd.read_excel(r"data.xlsx")

# 读取数据特征(126*16)
X = np.array(data.iloc[:,0:2])

# 读取数据标签(126*1)
Y1 = np.array(data.iloc[:,2:3])  # 土豆
Y2 = np.array(data.iloc[:,3:4])  # 西红柿
Y3 = np.array(data.iloc[:,4:5])  # 黄瓜
Y4 = np.array(data.iloc[:,5:6])  # 青椒

# # 验证集 测试集 数据划分
# X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=1.0, random_state=42)

# SVM模型训练
rfr = RandomForestRegressor(n_estimators=70,max_depth=1)
# 数据拟合
rfr.fit(X, Y1)

# 计算过往数据预测的MSE
print("模型对历史数据预测MSE为：", rfr.score(X,Y1))

# 测试集测试(有新数据替换X_text即可,y_pred即为预测数据的标签值)
# 预测未来5个月的数据
X_test = np.array([[2021,2021,2021,2021,2021],[2,3,4,5,6]]).reshape(5,2)
y_pred = rfr.predict(X_test)
print(f'未来{len(X_test)}个月土豆的销量为:{y_pred}')


# 模型保存
print("模型已经保存至rfr.pkl中")
with open("rfr.pkl", "wb") as f:
    pickle.dump(rfr, f)



# #----------------调参示范(max_depth)---------------------#
temp = []
for i in range(1,20,2):
    rfr = RandomForestRegressor(n_estimators=100, max_depth=i)
    rfr.fit(X, Y1)
    y_pred = rfr.predict(X_test)
    score = rfr.score(X,Y1)
    temp.append(score)
print(f"最好的max_depth取值为:max_depth={range(1,20,2)[temp.index(min(temp))]}")
plt.plot(temp,"-.")
plt.ylabel("MSE")
plt.xlabel("max_depth")
plt.show()
#
#
#
#----------------调参示范(n_estimators)---------------------#
temp = []
for i in range(50,200,10):
    rfr = RandomForestRegressor(n_estimators=i, max_depth=9)
    rfr.fit(X, Y1)
    y_pred = rfr.predict(X_test)
    score = rfr.score(X,Y1)
    temp.append(score)
x = list(range(50,200,10))
print(f"最好的n_estimators取值为:n_estimators={x[temp.index(min(temp))]}")
plt.plot(x,temp,"-.")
plt.ylabel("R2")
plt.xlabel("n_estimators")
plt.show()

