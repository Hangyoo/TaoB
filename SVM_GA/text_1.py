import numpy as np
import geatpy as ea
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc
import warnings
import pandas as pd
import numpy as np
import random


np.random.seed(100)
warnings.filterwarnings("ignore")

# 加载数据集
data = pd.read_csv("./data.csv")
X = np.array(data.iloc[:,:256])   # 读取数据集特征
y = np.array(data.iloc[:,-1]) # 读取数据集标签
print(f'数据集中包含的类别分别为(3分类问题)：',set(y))

# 对数据进行标准化
ss_x = StandardScaler()
X = ss_x.fit_transform(pd.DataFrame(X))

# 样本数， 特征数
n_samples, n_features = X.shape

# 训练样本的类别数量
n_classes = 3

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print('测试集规模：',X_train.shape, y_train.shape)
print('训练集规模：',X_test.shape, y_test.shape)

clf = svm.SVC(kernel='poly', C=6.6769, gamma=0.37588, degree=3.0, decision_function_shape='ovr',random_state=0)
# 在训练集上训练模型
clf.fit(X_train, y_train)
# 在测试集上测试模型
acc_score = clf.score(X_test, y_test)

# y_test = label_binarize(y_test, classes=clf.classes_)
print(f'分类准确率Acc = ：{acc_score*100}%')
