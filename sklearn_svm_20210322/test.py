import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

'''
用训练好的模型预测s3中数据(先运行main文件)
'''

# 加载保存的模型
with open(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\svm.pkl","rb") as f1:
    svc = pickle.load(f1)

# 防止警告，报错
warnings.filterwarnings("ignore")

# 正样本
data_acp_s3 = pd.read_csv(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\data\acp_s3.txt",header=None)
# 添加标签
data_acp_s3["400"] = [1]*(data_acp_s3.shape[0])


# 负样本
data_no_acp_s3 = pd.read_csv(r"C:\Users\Hangyu\PycharmProjects\TaoB\sklearn_svm\data\no_acp_s3.txt",header=None)
# 添加标签
data_no_acp_s3["400"] = [-1]*(data_no_acp_s3.shape[0])


#样本合并
data = pd.concat([data_acp_s3,data_no_acp_s3],axis=0)

# 读取数据特征
X = np.array(data.iloc[:,0:400])

# 读取数据标签
Y = np.array(data.iloc[:,400:401])

# 验证集 测试集 数据划分
_, X_test, _, y_test = train_test_split(X, Y, test_size=0.99,random_state=42)

# 预测
y_pred = svc.NN_predict(X_test)

# 预测准确率
score = svc.score(X_test,y_test)
print("准确率为:{}%".format(score*100))