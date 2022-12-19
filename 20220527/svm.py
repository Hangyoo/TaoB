import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# 加载数据
data = pd.read_csv(".\data.csv").iloc[:,7:]
# 读取数据特征(9*3)
data_inputs = np.array(data.iloc[:,:-1])
# 读取数据标签(144*2)
f1 = np.array(data.iloc[:,-1])

pca = PCA(n_components=10)  # 降维的个数
pca.fit(data_inputs,f1)
data_inputs = pca.transform(data_inputs)

ss_x = StandardScaler() # 实例化用于对特征标准化类
ss_y = StandardScaler() # 实例化用于对标签标准化类
# 对数据进行标准化
data_inputs = ss_x.fit_transform(pd.DataFrame(data_inputs))
# 读取数据标签
f1 = ss_y.fit_transform(pd.DataFrame(f1))

# PCA降维
# pca = PCA(n_components=10)  # 降维的个数
# pca.fit(data_inputs,f1)
# data_inputs = pca.transform(data_inputs)

#属性explained_variance_，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_)
#属性explained_variance_ratio_，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比，又叫做可解释方差贡献率
print(pca.explained_variance_ratio_)
# 总贡献率
print(pca.explained_variance_ratio_.sum())


# 验证集 测试集 数据划分
X_train, X_test, y_train, y_test = train_test_split(data_inputs, f1, test_size=0.7,random_state=42)

# 采用支持向量机分类
model = SVC()

model.fit(X_train, y_train.astype("int"))
score_train = model.score(X_train, y_train.astype("int"))
score_test = model.score(X_test, y_test.astype("int"))

print(f"模型在训练集上的精度:{score_train}")
print(f"模型在测试集上的精度:{score_test}")

