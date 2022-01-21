import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

'''
训练svm模型(先运行该文件)
'''

# 防止警告，报错
warnings.filterwarnings("ignore")

# 正样本
data_acp_s1 = pd.read_csv(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\data\acp_s1.txt",header=None)
data_acp_s2 = pd.read_csv(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\data\acp_s2.txt",header=None)
data_acp_s3 = pd.read_csv(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\data\acp_s3.txt",header=None)
# 添加标签
data_acp_s1["400"] = [1]*(data_acp_s1.shape[0])
data_acp_s2["400"] = [1]*(data_acp_s2.shape[0])
data_acp_s3["400"] = [1]*(data_acp_s3.shape[0])

# 负样本
data_no_acp_s1 = pd.read_csv(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\data\no_acp_s1.txt",header=None)
data_no_acp_s2 = pd.read_csv(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\data\no_acp_s2.txt",header=None)
data_no_acp_s3 = pd.read_csv(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\data\no_acp_s3.txt",header=None)
# 添加标签
data_no_acp_s1["400"] = [-1]*(data_no_acp_s1.shape[0])
data_no_acp_s2["400"] = [-1]*(data_no_acp_s2.shape[0])
data_no_acp_s3["400"] = [-1]*(data_no_acp_s3.shape[0])

#样本合并
data = pd.concat([data_acp_s1,data_no_acp_s1,data_acp_s2,data_no_acp_s2,data_acp_s3,data_no_acp_s3],axis=0)

# 读取数据特征
X = np.array(data.iloc[:,0:400])

# 读取数据标签
Y = np.array(data.iloc[:,400:401])

# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.85,random_state=42)

# SVM模型训练
svc = SVC(kernel="rbf")

# 数据拟合
svc.fit(X_train, y_train)

# 预测
y_pred = svc.predict(X_test)

# 预测准确率
score = svc.score(X_test,y_test)
print(f"准确率为:{score}")

print("模型已经保存至svm.pkl中")

# 模型保存
with open("svm.pkl", "wb") as f:
    pickle.dump(svc, f)
