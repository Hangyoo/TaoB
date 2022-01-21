import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm_regression\data.xlsx",index_col=0)

# 读取数据特征(126*16)
X = np.array(data.iloc[:,0:16])

# 读取数据标签(126*1)
Y = np.array(data.iloc[:,16:17])

# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.85,random_state=42)

# SVM模型训练
svr = SVR(kernel="rbf")

# 数据拟合
svr.fit(X_train, y_train)

# 测试集测试(有新数据替换X_text即可,y_pred即为预测数据的标签值)
y_pred = svr.predict(X_test)

print("模型已经保存至svm.pkl中")

# 模型保存
with open("svm.pkl", "wb") as f:
    pickle.dump(svr, f)