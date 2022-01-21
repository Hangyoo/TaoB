import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

'''
训练随机深林模型, 最重要的是设定随机树的个数

根据运行结果，最终设定 n_estimator = 200
'''

# 防止警告，报错
warnings.filterwarnings("ignore")

R2 = []

ES = [i for i in range(50,210,10)]
for i in ES:
    print(f'剩余评价次数：{len(ES) - ES.index(i)}')
    # 在这里导入数据文件
    data = pd.read_excel("data.xlsx",header=None)
    data = data.drop(index=0)

    # 在这里输出想保留特征的数目
    num = 60

    # 特征 和 label
    X = data.iloc[:,:247]
    Y = data.iloc[:,247]

    # 打乱数据集
    x_train, x_test, y_train, y_text = train_test_split(X, Y, random_state=21, test_size=0.8)

    # 建立随机深林模型
    rf = RandomForestRegressor(n_estimators=i)
    rf.fit(x_train,y_train)
    # r2 越大代表模型越好
    r2 = rf.score(x_test,y_text)
    R2.append(r2)

print(f"R2最优下，随机树个数为:{ES[R2.index(max(R2))]}")
plt.plot(ES, R2)
plt.ylabel("R2")
plt.xlabel("Number of estimator")
plt.show()
