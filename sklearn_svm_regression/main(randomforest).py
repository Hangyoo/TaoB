import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

data = pd.read_excel(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm_regression\data.xlsx",index_col=0)

# 读取数据特征(126*16)
X = np.array(data.iloc[:,0:15])

# 读取数据标签(126*1)
Y = np.array(data.iloc[:,16:17])

# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.85,random_state=42)

# SVM模型训练
rfr = RandomForestRegressor(n_estimators=60,max_depth=4)

# 数据拟合
rfr.fit(X_train, y_train)

# 测试集测试(有新数据替换X_text即可,y_pred即为预测数据的标签值)
y_pred = rfr.predict(X_test)

print("模型已经保存至rfr.pkl中")

# 模型保存
with open("rfr.pkl", "wb") as f:
    pickle.dump(rfr, f)

#----------------调参示范(max_depth)---------------------#
temp = []
for i in range(1,20,2):
    rfr = RandomForestRegressor(n_estimators=100, max_depth=i)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    score = r2_score(y_test,y_pred)
    temp.append(score)
print(f"最好的max_depth取值为:max_depth={temp.index(max(temp))}")
plt.plot(temp,"-.")
plt.ylabel("R2")
plt.xlabel("max_depth")
plt.show()



#----------------调参示范(n_estimators)---------------------#
temp = []
for i in range(50,200,10):
    rfr = RandomForestRegressor(n_estimators=i, max_depth=9)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    score = r2_score(y_test,y_pred)
    temp.append(score)
x = list(range(50,200,10))
print(f"最好的n_estimators取值为:max_depth={x[temp.index(max(temp))]}")
plt.plot(x,temp,"-.")
plt.ylabel("R2")
plt.xlabel("n_estimators")
plt.show()

