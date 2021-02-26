from sklearn.cluster import OPTICS, cluster_optics_dbscan
import numpy as np
import pandas as pd
import random

# 生成随机数据
# data = [[random.randint(0,50) for i in range(20)] for j in range(50)]
# data = np.array(data)
# data = pd.DataFrame(data)
# data.to_excel("data.xls",header=False)
# print(data.shape)

# n_points_per_cluster = 50
# C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
# C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
# C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
# C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
# C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
# C7 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
# C8 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
# C9 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
# C10 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
# C11 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C12 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
# C13 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
# C14 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
# C15 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
# C16 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
# C17 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C18 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
# C19 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
# C20 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
# X = np.hstack((C1, C2, C3, C4, C5, C6, C7, C8, C9, C10))
# print(X.shape)

#数据加载
X = pd.read_excel("data.xls")

# 聚类算法参数设置(最小聚类数目)
clust = OPTICS(min_cluster_size=5)

# 拟合
labels = clust.fit_predict(X)

# 类别估计
# labels = cluster_optics_dbscan(reachability=clust.reachability_,core_distances=clust.core_distances_,ordering=clust.ordering_, eps=0.5)

# 类别输出
print(labels)
