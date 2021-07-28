from sklearn.cluster import Birch
import pandas as pd

#数据加载
X = pd.read_excel("data.xls")

# 聚类算法参数设置(最小聚类数目)
# 改变n_clusters=5可以修改指定的聚类数目
# threshold:即叶节点每个CF的最大样本半径阈值T，它决定了每个CF里所有样本形成的超球体的半径阈值（设置为0.5较好）
# 默认是50。如果样本量非常大，比如大于10万，则一般需要增大这个默认值
birch = Birch(threshold=0.5, branching_factor=50, n_clusters=5,
                 compute_labels=True, copy=True)
# 拟合
model = birch.fit(X)
# 类别估计
labels = model.NN_predict(X)

# 类别输出
print(labels)