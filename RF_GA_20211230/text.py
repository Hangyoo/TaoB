import numpy as np
import pickle
import matplotlib.pyplot as plt



Val1 = 1.0 # 默认钻头为牙轮
Val2 = 0.142857 # 默认岩性为泥岩
max_val1,min_val1 = 3, 0
max_val2,min_val2 = 7, 0
max_depth, min_depth = 7655, 500


# 读取训练好的机器学习模型
with open("RF.pkl", "rb") as f1:  # 预测loss
    model1 = pickle.load(f1)


depth = 7650  # 速度3.935
x1,x2,x3,x4,x5,x6,x7,x8 = 241.3,8.08,59,16.4,21.6,1.50,50.094,39.131
X = np.array([depth, x1, Val1, Val2, x2, x3, x4, x5, x6,x7,x8]).reshape(1, 11)
f1 = model1.predict(X)
print(f1)

depth = 7650  # 速度17.71
x1,x2,x3,x4,x5,x6,x7,x8 = 305.75,0,35,3676.33,3.999,1.0,79,91.13
X = np.array([depth, x1, Val1, Val2, x2, x3, x4, x5, x6,x7,x8]).reshape(1, 11)
# 计算目标函数值
f1 = model1.predict(X)
print(f1)
